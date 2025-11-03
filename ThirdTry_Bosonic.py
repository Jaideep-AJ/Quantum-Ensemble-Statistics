import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


# Physical constants
KB = 1.3806452E-23  # Boltzmann constant
h = 6.636e-34  # Planck constant

# Choose your boson type:
# Option 1: Helium-4 atoms (realistic BEC experiments)
mass = 6.646e-27  # Helium-4 atom mass in kg

# Option 2: Electron mass (lighter, for faster simulation)
# mass = 9.1e-31  # Electron mass

L = 5e-9  # Box length 5 nm


# Energy scale: E0 = h^2/(8*m*L^2)
E0 = h**2 / (8 * mass * L**2)


# Simulation parameters
N = 2000
T = 200  # Temperature in Kelvin (lowered for stronger quantum effects)
steps_to_equilibrium = 50000
steps_to_explore = 150000
MAX_QUANTUM_NUMBER = 13
debug = False


print(f"Energy scale E0 = {E0:.6e} J")
print(f"Thermal energy kB*T = {KB*T:.6e} J")
print(f"Ratio E0/(kB*T) = {E0/(KB*T):.6f}")
print(f"Particle type: {'Helium-4' if mass > 1e-26 else 'Electron-like'}")


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
        for l in range(1, MAX_QUANTUM_NUMBER + 1):
            for m in range(1, MAX_QUANTUM_NUMBER + 1):
                for n in range(1, MAX_QUANTUM_NUMBER + 1):
                    E_tilde = l**2 + m**2 + n**2
                    self.degeneracy[E_tilde] = self.degeneracy.get(E_tilde, 0) + 1

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
        id = 0
        for l in range(1, MAX_QUANTUM_NUMBER+1):
            for m in range(1, MAX_QUANTUM_NUMBER+1):
                for n in range(1, MAX_QUANTUM_NUMBER+1):
                    # Only one particle per quantum state initially
                    # (no spin degeneracy for bosons)
                    particle = Particle(l, m, n, id)
                    id += 1
                    self.system.add_particle(particle)

                    if id >= N:
                        return

    def walk(self, n_trials, record_interval=200):
        """Take `n_trials` random walks in the gamma space
        `record_interval` must be > 0 for recording stats"""

        accepted, rejected = 0, 0
        occupation_numbers = defaultdict(int)
        n_samples = 0

        for i in tqdm(range(n_trials)):
            chosen_one = np.random.randint(0, N)
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
                    occupation_numbers[energy] += n_particles
                n_samples += 1

        # Calculate average occupation per quantum state
        average_occupation_per_energy = {}
        for energy, n_particles in occupation_numbers.items():
            degen = self.system.degeneracy.get(energy, 1)
            average_occupation_per_energy[energy] = n_particles/(n_samples*degen)

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


def find_mu_from_simulation(avg_occ, degeneracy_map, N, T_tilde):
    """
    Find chemical potential that matches simulation particle number

    Parameters:
    -----------
    avg_occ : dict
        Average occupation per quantum state from simulation {E_tilde: n_avg}
    degeneracy_map : dict
        Degeneracy for each energy level {E_tilde: g}
    N : int
        Total particle number
    T_tilde : float
        Normalized temperature kB*T/E0

    Returns:
    --------
    mu_tilde : float
        Normalized chemical potential mu/E0
    """
    energies = np.array(sorted(avg_occ.keys()))
    n_sim = np.array([avg_occ[e] for e in energies])
    g_vals = np.array([degeneracy_map[e] for e in energies], dtype=float)

    # Total particles from simulation (should equal N)
    N_check = np.sum(g_vals * n_sim)
    print(f"Simulation particle count check: {N_check:.1f} (expected {N})")

    # Find μ_tilde that reproduces this N using BE distribution
    def total_particles(mu_tilde):
        # BE distribution
        exponent = (energies - mu_tilde) / T_tilde
        # Avoid numerical issues
        exponent = np.clip(exponent, 0.01, 100)
        n_BE = 1.0 / (np.exp(exponent) - 1.0)
        return np.sum(g_vals * n_BE)

    # Bosons: mu must be less than ground state energy
    E_ground = np.min(energies)
    mu_high = E_ground - 1e-6
    mu_low = E_ground - 100 * T_tilde

    # Bisection search
    for iteration in range(100):
        mu_mid = 0.5 * (mu_low + mu_high)
        N_mid = total_particles(mu_mid)
        if N_mid > N:
            mu_high = mu_mid
        else:
            mu_low = mu_mid

        if abs(N_mid - N) < 0.01 * N:
            break

    mu_tilde = 0.5 * (mu_low + mu_high)
    print(f"Converged after {iteration+1} iterations")
    return mu_tilde


def calculate_BE_distribution(energy_array, mu_tilde, T_tilde):
    """
    Calculate Bose-Einstein distribution for given energy array

    Parameters:
    -----------
    energy_array : array-like
        Energy values (normalized)
    mu_tilde : float
        Chemical potential (normalized)
    T_tilde : float
        Temperature (normalized)

    Returns:
    --------
    be_occupation : array
        Bose-Einstein occupation numbers
    """
    be_occupation = []
    for e_tilde in energy_array:
        exponent = (e_tilde - mu_tilde) / T_tilde
        # Avoid numerical issues
        if exponent > 0.01:
            f_BE = 1.0 / (np.exp(exponent) - 1.0)
        else:
            # Near condensation - handle carefully
            f_BE = T_tilde / (e_tilde - mu_tilde) if (e_tilde - mu_tilde) > 0 else 1e6
        be_occupation.append(f_BE)
    return np.array(be_occupation)


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
    print("\nEquilibriating...")
    ens.walk(steps_to_equilibrium, record_interval=-1)
    print()
    print(f"Total normalized energy after equilibration: {ens.system.E_total}")
    print(f"Average energy per particle: {ens.system.E_total/N:.2f}")

    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel(r'Normalized Energy $\tilde{E} = l^2 + m^2 + n^2$')
    plt.ylabel('Number of Particles')
    plt.title('Instantaneous Boson Distribution (After Equilibration)')
    plt.show()

    # Production run with recording
    print("\nExploring the gamma space...")
    avg_occ = ens.walk(steps_to_explore, record_interval=100)
    print()
    print(f"Final total normalized energy: {ens.system.E_total}")
    print(f"Average energy per particle: {ens.system.E_total/N:.2f}")

    energy = sorted(avg_occ.keys())  # normalized energies
    avg_per_energy = [avg_occ[e] for e in energy]  # average occupation per state

    # === COMPARISON WITH BOSE-EINSTEIN STATISTICS ===

    # Normalized temperature
    T_tilde = KB * T / E0
    print(f"\n{'='*50}")
    print(f"=== Bose-Einstein Comparison ===")
    print(f"{'='*50}")
    print(f"Normalized temperature kBT/E0 = {T_tilde:.6f}")

    # Extract chemical potential from simulation
    mu_tilde = find_mu_from_simulation(avg_occ, ens.system.degeneracy, N, T_tilde)
    print(f"Chemical potential μ/E0 = {mu_tilde:.6f}")
    print(f"Ground state energy E_ground/E0 = {min(energy)}")
    print(f"Difference (E_ground - μ)/E0 = {min(energy) - mu_tilde:.6f}")

    # Check for condensation
    ground_state_occupation = avg_occ.get(min(energy), 0)
    condensate_fraction = ground_state_occupation / N * ens.system.degeneracy.get(min(energy), 1)
    print(f"\nGround state occupation: {ground_state_occupation:.2f}")
    print(f"Condensate fraction: {condensate_fraction*100:.2f}%")
    if condensate_fraction > 0.1:
        print("*** SIGNIFICANT CONDENSATION DETECTED ***")

    # Generate theoretical BE distribution at discrete levels
    be_f_of_E = calculate_BE_distribution(energy, mu_tilde, T_tilde)

    # Generate smooth BE curve for better visualization
    energy_fine = np.linspace(min(energy), max(energy), 500)
    be_f_fine = calculate_BE_distribution(energy_fine, mu_tilde, T_tilde)

    # === PLOT 1: Linear scale with smooth curve ===
    plt.figure(figsize=(12, 7))
    plt.scatter(energy, avg_per_energy, label='MC Simulation', 
                s=100, alpha=0.7, color='blue', zorder=3, edgecolors='darkblue')
    plt.plot(energy_fine, be_f_fine, label='Bose-Einstein (smooth)', 
             color='red', linewidth=2.5, linestyle='-', zorder=1)
    plt.scatter(energy, be_f_of_E, label='BE at discrete levels', 
                s=60, alpha=0.6, color='orange', marker='x', zorder=2, linewidths=2)
    plt.xlabel(r'Normalized Energy $\tilde{E} = l^2 + m^2 + n^2$', fontsize=13)
    plt.ylabel('Average Occupation per Quantum State', fontsize=13)
    plt.title(f'Boson Distribution with Smooth BE Curve\n(T={T}K, N={N}, $\tilde{{T}}$={T_tilde:.3f})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # === PLOT 2: Log scale ===
    plt.figure(figsize=(12, 7))
    plt.scatter(energy, avg_per_energy, label='MC Simulation', 
                s=100, alpha=0.7, color='blue', zorder=3, edgecolors='darkblue')
    plt.plot(energy_fine, be_f_fine, label='Bose-Einstein (smooth)', 
             color='red', linewidth=2.5, linestyle='-', zorder=1)
    plt.scatter(energy, be_f_of_E, label='BE at discrete levels', 
                s=60, alpha=0.6, color='orange', marker='x', zorder=2, linewidths=2)
    plt.xlabel(r'Normalized Energy $\tilde{E} = l^2 + m^2 + n^2$', fontsize=13)
    plt.ylabel('Average Occupation per Quantum State (log scale)', fontsize=13)
    plt.title(f'Boson Distribution - Log Scale\n(T={T}K, N={N}, $\tilde{{T}}$={T_tilde:.3f})', 
              fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

    # === PLOT 3: Residual plot ===
    residuals = []
    for e_tilde, n_sim, n_be in zip(energy, avg_per_energy, be_f_of_E):
        if n_be > 1e-10:  # Avoid division by very small numbers
            rel_error = (n_sim - n_be) / n_be * 100
            residuals.append(rel_error)
        else:
            residuals.append(0)

    plt.figure(figsize=(12, 6))
    plt.scatter(energy, residuals, s=80, alpha=0.7, color='purple', edgecolors='darkviolet')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.xlabel(r'Normalized Energy $\tilde{E} = l^2 + m^2 + n^2$', fontsize=13)
    plt.ylabel('Relative Error (%)', fontsize=13)
    plt.title('Residual Analysis: (Simulation - Theory) / Theory × 100%', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print residual statistics
    residuals_clean = [r for r in residuals if abs(r) < 1000]  # Remove outliers
    print(f"\n{'='*50}")
    print(f"=== Statistical Analysis ===")
    print(f"{'='*50}")
    print(f"Mean relative error: {np.mean(residuals_clean):.2f}%")
    print(f"Standard deviation: {np.std(residuals_clean):.2f}%")
    print(f"Max relative error: {np.max(np.abs(residuals_clean)):.2f}%")
    print(f"\nNote: Lower temperatures will show better agreement and")
    print(f"stronger quantum effects including potential condensation.")


if __name__ == '__main__':
    main()
