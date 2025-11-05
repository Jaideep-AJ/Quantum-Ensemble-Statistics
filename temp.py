import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from scipy.optimize import brentq # Added for chemical potential calculation

# Physical constants
KB = 1.3806452E-23  # Boltzmann constant
h = 6.636e-34  # Planck constant
mass = 9.1e-31  # Electron mass
L = 5e-9  # Box length 5 nm

# Energy scale: E0 = h^2/(8*m*L^2)
E0 = h**2 / (8 * mass * L**2)

# Simulation parameters
N = 2000
T = 300  # Temperature in Kelvin
steps_to_equilibrium = 50000
steps_to_explore = 150000
MAX_QUANTUM_NUMBER = 13
debug = False

print(f"Energy scale E0 = {E0:.6e} J")
print(f"Thermal energy kB*T = {KB*T:.6e} J")
print(f"Ratio E0/(kB*T) = {E0/(KB*T):.6f}")

class Particle():
    """
    Create a Particle in a 3D box
    ----
    PARAMETERS
    ----
    l : wave vector in x
    m : wave vector in y
    n : wave vector in z
    index : particle identifier "name"

    NOTE: Bosons have no spin degeneracy, each quantum state (l,m,n) is unique
    Energy is normalized: E_tilde = l^2 + m^2 + n^2
    """
    def __init__(self, l, m, n, index):
        self.k = (l, m, n)
        self.E = 0  # Normalized energy E_tilde
        self.update_energy()
        self.index = index

    def update_energy(self):
        """Updates value of normalized energy of particle"""
        # Normalized energy: E_tilde = l^2 + m^2 + n^2
        self.E = self.k[0]**2 + self.k[1]**2 + self.k[2]**2

    def __eq__(self, other):
        """Check equivalence between particles"""
        try:
            if self.k[0] == other.k[0] and self.k[1] == other.k[1] and self.k[2] == other.k[2]:
                return True
            else:
                return False
        except Exception:
            return False

class Microstate():
    """
    Create a certain configuration of N particles in a 3D box
    ----
    PARAMETERS
    ----
    N : Number of particles
    """
    def __init__(self, N):
        self.particles_hashmap = {}  # {(l,m,n): [particle_indices]}
        self.particles = {}  # {index: Particle}
        self.energy_map = {}  # {E_tilde: occupation_number}
        self.E_total = 0  # Total normalized energy

        # Compute degeneracy map using normalized energy
        self.degeneracy = {}
        # Ensure we cover all possible E_tilde values up to MAX_QUANTUM_NUMBER for degeneracy calculation
        # It's better to pre-calculate all possible E_tilde values and their degeneracies here
        # and store them in a sorted list for later use with chemical potential calculation.
        all_possible_energies = set()
        for l in range(1, MAX_QUANTUM_NUMBER + 1):
            for m in range(1, MAX_QUANTUM_NUMBER + 1):
                for n in range(1, MAX_QUANTUM_NUMBER + 1):
                    E_tilde = l**2 + m**2 + n**2
                    self.degeneracy[E_tilde] = self.degeneracy.get(E_tilde, 0) + 1
                    all_possible_energies.add(E_tilde)
        self.all_possible_energies_sorted = sorted(list(all_possible_energies))


    def add_particle(self, state: Particle):
        """Add a particle to the ensemble while building it.
        Returns 0 if addition successful.

        NOTE: For bosons, there is NO restriction on occupation number!
        Multiple bosons can occupy the same quantum state."""

        if state.k in self.particles_hashmap:
            # Add boson to the state (no limit for bosons!)
            self.particles_hashmap[state.k].append(state.index)
            self.particles[state.index] = state
            self.energy_map[state.E] += 1
        else:
            # Create new entry for this quantum state
            self.particles_hashmap[state.k] = [state.index]
            self.particles[state.index] = state
            if state.E in self.energy_map:
                self.energy_map[state.E] += 1
            else:
                self.energy_map[state.E] = 1

        # Update microstate energy
        self.E_total += state.E
        return 0

    def transition(self, particle_index: int, new_state: tuple):
        """
        Execute a particle swap. This changes the energy map of the system
        Returns
        1 if swap accepted
        0 if swap rejected

        NOTE: For bosons, NO transitions are forbidden due to Pauli exclusion!
        Uses normalized energy for Metropolis criterion.
        """

        # Get the particle to transition
        concerned_particle = self.particles[particle_index]

        # Calculate new normalized energy
        new_particle = Particle(new_state[0], new_state[1], new_state[2],
                               concerned_particle.index)
        new_energy = new_particle.E
        old_energy = concerned_particle.E
        delta_E = new_energy - old_energy

        # Metropolis acceptance criterion (using normalized energy)
        # Actual energy difference: delta_E_actual = delta_E * E0
        coin = np.random.random()
        to_accept = delta_E <= 0 or coin < np.exp(-delta_E * E0 / (KB * T))

        if debug:
            print(f'delta_E = {delta_E}; exp = {np.exp(-delta_E * E0 / (KB * T))}; coin={coin}')
            print(f'{"Accep" if to_accept else "Rejec"}ting jump')

        if to_accept:
            old_k = concerned_particle.k
            old_E = concerned_particle.E

            # Update energy maps
            self.energy_map[old_E] -= 1
            if self.energy_map[old_E] == 0:
                del self.energy_map[old_E]
            self.E_total -= old_E

            # Remove from old state in hashmap
            self.particles_hashmap[old_k].remove(concerned_particle.index)
            if len(self.particles_hashmap[old_k]) == 0:
                del self.particles_hashmap[old_k]

            # Update particle state
            concerned_particle.k = new_state
            concerned_particle.update_energy()

            # Add to new state in hashmap
            if new_state in self.particles_hashmap:
                self.particles_hashmap[new_state].append(concerned_particle.index)
            else:
                self.particles_hashmap[new_state] = [concerned_particle.index]

            # Update energy maps
            if concerned_particle.E in self.energy_map:
                self.energy_map[concerned_particle.E] += 1
            else:
                self.energy_map[concerned_particle.E] = 1

            self.E_total += concerned_particle.E
            return 1
        else:
            return 0

    def get_stats(self):
        """Get energy distribution E_tilde vs n dictionary"""
        return self.energy_map

    def print_stats(self):
        print("QUANTUM NUMBERS\tOCCUPATION NUMBER")
        print("____________________________________________")
        for k in self.particles_hashmap:
            print(f"{k}\t{len(self.particles_hashmap[k])}")
        print("____________________________________________")

class Ensemble():
    def __init__(self, N):
        self.N = N
        self.system = Microstate(N)

    def build_ensemble(self):
        """Builds a system of N bosons"""
        # Start by filling the lowest energy states
        id = 0
        energies_sorted = sorted(self.system.degeneracy.keys())
        
        # Populate particles in the lowest energy states first until N particles are added
        # This is a more reasonable initial state than randomly picking
        for E_tilde in energies_sorted:
            degeneracy_at_E = self.system.degeneracy[E_tilde]
            for _ in range(degeneracy_at_E): # Each quantum state can hold many bosons
                if id < self.N:
                    # Find a k-state for this E_tilde (simple example, you might need a better way if N is very large)
                    # For simplicity, let's just add to the current lowest energy state available
                    # A better way would be to iterate through (l,m,n) explicitly to get distinct states
                    # For a quick fix, let's assume we can find an l,m,n that gives this E_tilde
                    # This initial population strategy is still a bit crude but better than random for bosons.
                    # The simulation should equilibrate it anyway.
                    
                    # Instead of finding specific (l,m,n) for initial state,
                    # just assign particles to (1,1,1) if it exists, otherwise to any lowest
                    # This is simpler and the MC will redistribute.
                    
                    # Let's simplify and just put them all into the ground state (1,1,1) initially
                    # This makes sense for bosons as it's the lowest energy configuration.
                    particle = Particle(1, 1, 1, id) # All N particles start in the ground state (1,1,1)
                    self.system.add_particle(particle)
                    id += 1
                else:
                    break
            if id >= self.N:
                break
        
        # If N is larger than the number of available (l,m,n) states up to MAX_QUANTUM_NUMBER,
        # the initial state might not fill all of them unique for other types of particles.
        # But for bosons, they can all crowd into (1,1,1).
        # So we just ensure all N particles are created.
        while id < self.N:
            particle = Particle(1, 1, 1, id)
            self.system.add_particle(particle)
            id += 1


    def walk(self, n_trials, record_interval=200):
        """Take `n_trials` random walks in the gamma space
        `record_interval` must be > 0 for recording stats"""

        accepted, rejected = 0, 0
        occupation_numbers = defaultdict(int)
        n_samples = 0

        for i in tqdm(range(n_trials)):
            chosen_one = np.random.randint(0, self.N) # Use self.N, not N from global scope
            new_l = np.random.randint(1, MAX_QUANTUM_NUMBER + 1)
            new_m = np.random.randint(1, MAX_QUANTUM_NUMBER + 1)
            new_n = np.random.randint(1, MAX_QUANTUM_NUMBER + 1)
            new_quantum_numbers = (new_l, new_m, new_n)

            result = self.system.transition(chosen_one, new_quantum_numbers)

            if result == 1:
                accepted += 1
            else:
                rejected += 1

            if record_interval > 0 and i % record_interval == 0:
                for energy, n_particles in self.system.energy_map.items():
                    # occupation_numbers[energy] stores total particles encountered at this energy level
                    # over all samples.
                    occupation_numbers[energy] += n_particles
                n_samples += 1

        # Calculate average occupation per quantum state
        average_occupation_per_energy = {}
        for energy, total_n_particles_at_E in occupation_numbers.items():
            degen = self.system.degeneracy.get(energy, 1)
            # The average occupation of a *state* (not an energy level)
            # is total_n_particles_at_E / (n_samples * degen)
            # This is correct for comparison with f(E) from BE distribution.
            average_occupation_per_energy[energy] = total_n_particles_at_E / (n_samples * degen)

        self.stats = {
            "n_trials": n_trials,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": round(accepted / n_trials, 2),
            "rejectance_rate": round(rejected / n_trials, 2)
        }

        print("\n--- Simulation Summary ---")
        for k, v in self.stats.items():
            print(f"{k:20s}: {v}")

        return average_occupation_per_energy


def calculate_N(mu, T_val, energies_tilde, degeneracies, E0_val, KB_val):
    """Calculates the total number of particles for a given chemical potential mu."""
    total_N_calc = 0
    for e_tilde in energies_tilde:
        energy_actual = e_tilde * E0_val
        degen = degeneracies.get(e_tilde, 0)
        
        # Avoid division by zero or negative argument for log if mu is too high
        if energy_actual - mu < 0 and degen > 0:
             # This indicates a problem with mu, or system parameters leading to condensation.
             # For a simple approach, assume mu < E_min
             # More robust solutions might involve handling the ground state separately.
             # If mu is above the ground state, BE distribution definition breaks down
             # for the ground state.
             return float('inf') # Signal that mu is too high
        
        exponent = (energy_actual - mu) / (KB_val * T_val)
        
        if degen > 0 and exponent < 100: # Avoid overflow in exp, treat large exponent as 0 occupation
            # For bosons, the denominator is (exp(...) - 1)
            # If exp(...) is very close to 1 (i.e., E-mu is small), this can lead to large occupations.
            if exponent <= 0: # This case shouldn't happen if mu < E_min (E_actual)
                # If E_actual == mu, then occupation is infinite.
                # If E_actual < mu, this is unphysical for bosons or means condensation.
                # Here, we will assume mu is always less than the ground state E_actual_min.
                # If it tries to go above, we can consider it an invalid mu.
                return float('inf') 
            
            total_N_calc += degen / (np.exp(exponent) - 1)
    return total_N_calc

def find_chemical_potential(N_target, T_val, energies_tilde, degeneracies, E0_val, KB_val):
    """Finds the chemical potential mu that yields N_target particles."""
    # The chemical potential mu for bosons must be less than the ground state energy.
    # The ground state energy is E_tilde_min * E0_val, where E_tilde_min = 1^2+1^2+1^2 = 3
    E_min_actual = min(energies_tilde) * E0_val
    
    # We need to find mu such that calculate_N(mu) = N_target
    # The function calculate_N is monotonically decreasing with mu.
    # So we can use a root-finding algorithm like brentq.
    
    # Define a function whose root is the desired mu
    def func_to_minimize(mu_val):
        return calculate_N(mu_val, T_val, energies_tilde, degeneracies, E0_val, KB_val) - N_target

    # Determine a bracket for mu.
    # Lower bound: mu must be less than E_min_actual
    # It can be slightly negative or very close to E_min_actual.
    # A safe lower bound would be something like E_min_actual - 100 * KB_val * T_val (very negative)
    # A safe upper bound for mu is slightly less than E_min_actual.
    
    # Let's try to find a bracket using trial values.
    # Lower bound for mu: A value far below E_min_actual, ensuring calculate_N is large.
    mu_lower_bound = E_min_actual - 50 * KB_val * T_val 
    
    # Upper bound for mu: A value very close to E_min_actual, ensuring calculate_N is small but finite.
    # Be careful not to make E_actual - mu_upper_bound too close to zero for the ground state.
    mu_upper_bound = E_min_actual - 1e-12 * KB_val * T_val # Must be strictly less than E_min_actual

    try:
        # Check signs at boundaries
        val_lower = func_to_minimize(mu_lower_bound)
        val_upper = func_to_minimize(mu_upper_bound)
        
        if np.sign(val_lower) == np.sign(val_upper):
            print(f"Warning: Root finding bracket has same sign for mu. val_lower={val_lower}, val_upper={val_upper}")
            print(f"Adjusting bracket bounds. E_min_actual={E_min_actual:.4e}, KB_val*T_val={KB_val*T_val:.4e}")
            # If they have the same sign, we need to adjust.
            # Usually means mu_lower_bound is not low enough (calculate_N is too low)
            # or mu_upper_bound is not high enough (calculate_N is too high)
            # Let's try a wider range for mu_lower_bound.
            # And ensure mu_upper_bound is really close to E_min_actual without crossing.
            
            # Try to bracket by searching
            test_mu_low = E_min_actual - 100 * KB_val * T_val
            while func_to_minimize(test_mu_low) < 0 and test_mu_low < (E_min_actual - 1e-10 * KB_val * T_val):
                test_mu_low += 10 * KB_val * T_val
            
            test_mu_high = E_min_actual - 1e-12 * KB_val * T_val
            while func_to_minimize(test_mu_high) > 0 and test_mu_high > (E_min_actual - 100 * KB_val * T_val):
                test_mu_high -= 10 * KB_val * T_val
            
            if np.sign(func_to_minimize(test_mu_low)) != np.sign(func_to_minimize(test_mu_high)):
                mu_lower_bound = test_mu_low
                mu_upper_bound = test_mu_high
            else:
                print("Could not find suitable bracket for mu, using a default.")
                # Fallback if dynamic bracketing fails
                mu_lower_bound = E_min_actual - 100 * KB_val * T_val 
                mu_upper_bound = E_min_actual - 1e-9 * KB_val * T_val
                if func_to_minimize(mu_lower_bound) > 0 and func_to_minimize(mu_upper_bound) < 0:
                    pass # Good bracket
                else:
                    print("Default bracket also problematic, trying a very wide range.")
                    mu_lower_bound = E_min_actual - 1000 * KB_val * T_val
                    mu_upper_bound = E_min_actual - 1e-15 * KB_val * T_val # Very close to E_min
                    if np.sign(func_to_minimize(mu_lower_bound)) == np.sign(func_to_minimize(mu_upper_bound)):
                        raise ValueError("Failed to find a valid bracket for chemical potential.")


        mu_found = brentq(func_to_minimize, mu_lower_bound, mu_upper_bound, xtol=1e-15)
        return mu_found
    except ValueError as e:
        print(f"Error finding chemical potential: {e}")
        print("Falling back to mu = 0 for theoretical plot, which may not be accurate.")
        return 0 # Fallback for now if root finding fails

def main():
    ens = Ensemble(N)
    ens.build_ensemble()

    print(f"\nTotal normalized energy: {ens.system.E_total}")
    print(f"Average energy per particle: {ens.system.E_total/N:.2f}")

    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel(r'Normalized Energy $\tilde{E} = l^2 + m^2 + n^2$')
    plt.ylabel('Number of Particles')
    plt.title('Boson Statistics (Initial)')
    plt.show()

    # Equilibration
    print("Equilibriating...")
    ens.walk(steps_to_equilibrium, record_interval=-1)
    print()
    print(f"\nTotal normalized energy after equilibration: {ens.system.E_total}")
    print(f"Average energy per particle: {ens.system.E_total/N:.2f}")

    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel(r'Normalized Energy $\tilde{E} = l^2 + m^2 + n^2$')
    plt.ylabel('Number of Particles')
    plt.title('Instantaneous Boson Distribution (Equilibrium)')
    plt.show()

    # Production run with recording
    print("Exploring the gamma space...")
    avg_occ = ens.walk(steps_to_explore, record_interval=100)
    print()
    print(f"\nFinal total normalized energy: {ens.system.E_total}")
    print(f"Average energy per particle: {ens.system.E_total/N:.2f}")

    # Sort energies for plotting
    sim_energies = sorted(avg_occ.keys())  # normalized energies
    avg_per_state_sim = [avg_occ[e] for e in sim_energies]  # average occupation per state

    plt.figure(figsize=(10, 6))
    plt.scatter(sim_energies, avg_per_state_sim, label='Simulation', s=50, alpha=0.7)
    plt.xlabel(r'Normalized Energy $\tilde{E} = l^2 + m^2 + n^2$')
    plt.ylabel('Average Occupation per Quantum State')
    plt.title(f'Boson Energy Distribution (T={T}K, N={N})')

    # Comparison with Bose-Einstein statistics
    # 1. Get all possible energies and their degeneracies from the Microstate
    all_energies_tilde_for_mu_calc = ens.system.all_possible_energies_sorted
    degeneracies_for_mu_calc = ens.system.degeneracy
    
    # 2. Find the chemical potential mu
    print("\nFinding chemical potential for theoretical Bose-Einstein distribution...")
    mu_theoretical = find_chemical_potential(N, T, all_energies_tilde_for_mu_calc, 
                                             degeneracies_for_mu_calc, E0, KB)
    
    print(f"Calculated Chemical Potential (μ): {mu_theoretical:.6e} J")
    print(f"μ / (KB*T): {mu_theoretical / (KB*T):.6f}")
    
    # 3. Calculate Bose-Einstein distribution with the found mu
    be_E_plot = []
    be_f_of_E_plot = []

    # Use a wider range of energies for the theoretical plot to show the full curve
    # including potential values beyond those sampled by simulation if MAX_QUANTUM_NUMBER allows.
    # However, for direct comparison, we should use the same energies as the simulation's results.
    # Or at least a superset covering sim_energies.
    
    # Let's use sim_energies for direct comparison
    for e_tilde in sim_energies:
        be_E_plot.append(e_tilde)
        
        energy_actual = e_tilde * E0
        exponent = (energy_actual - mu_theoretical) / (KB * T)
        
        # Guard against unphysical or problematic exponents
        if exponent <= 0: # This implies E <= mu, which shouldn't happen for valid mu < E_ground
            # For plotting, if we hit this, it means mu is too high, or E_actual is too low.
            # This is a critical point for Bose-Einstein condensation.
            # If E_actual is very close to mu, occupation can be very large.
            # If E_actual == mu, occupation is infinite.
            # For plotting, we might cap it or make it NaN/inf.
            f_of_E = float('inf') if exponent == 0 else 1000.0 # Cap to a large value for plot
        elif exponent > 100:  # Avoid overflow, f(E) ≈ 0
            f_of_E = 0
        else:
            f_of_E = 1 / (np.exp(exponent) - 1)
        be_f_of_E_plot.append(f_of_E)

    # Plot the theoretical Bose-Einstein distribution
    # No amplitude scaling is needed because both quantities (simulation and theoretical)
    # represent the 'average occupation per quantum state'.
    plt.plot(be_E_plot, be_f_of_E_plot, label='Bose-Einstein (Theoretical)', 
             color='r', linewidth=2, linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0) # Ensure occupation numbers are non-negative
    plt.show()

if __name__ == '__main__':
    main()