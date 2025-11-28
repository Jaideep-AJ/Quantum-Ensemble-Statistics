import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Physical constants
KB = 1.3806452E-23  # Boltzmann constant
h = 6.636e-34  # Planck constant
# mass = 9.1e-31  # Electron mass
mass = 1.44316060e-25  # Rubidium-87 mass (kg)
L = 5e-9  # Box length 5 nm

# Energy scale: E0 = h^2/(8*m*L^2)
E0 = h**2 / (8 * mass * L**2)

# Simulation parameters
N = 5000
T = 1  # Temperature in Kelvin
steps_to_equilibrium = 50000
steps_to_explore = 150000
MAX_QUANTUM_NUMBER = 40
debug = False
n_decimal_places = 24

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
    Energy is now PHYSICAL: E = E0 * (l^2 + m^2 + n^2), stored rounded to 25 dp.
    """
    def __init__(self, l, m, n, index):
        self.k = (l, m, n)
        self.E = 0  # Physical energy E (J), rounded
        self.update_energy()
        self.index = index

    def update_energy(self):
        """Updates physical energy of particle; stores rounded to 25 dp."""
        E_phys = E0 * (self.k[0]**2 + self.k[1]**2 + self.k[2]**2)
        self.E = round(E_phys, n_decimal_places)

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
        self.energy_map = {}  # {E (rounded): occupation_number}
        self.E_total = 0  # Total energy (sum of rounded energies)

        # Compute degeneracy map using PHYSICAL energy keys (rounded)
        self.degeneracy = {}
        for l in range(1, MAX_QUANTUM_NUMBER + 1):
            for m in range(1, MAX_QUANTUM_NUMBER + 1):
                for n in range(1, MAX_QUANTUM_NUMBER + 1):
                    E_phys = E0 * (l**2 + m**2 + n**2)
                    E_key = round(E_phys, n_decimal_places)
                    self.degeneracy[E_key] = self.degeneracy.get(E_key, 0) + 1

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
        Uses PHYSICAL energy for Metropolis criterion.
        """
        # Get the particle to transition
        concerned_particle = self.particles[particle_index]

        # Calculate new physical energy
        new_particle = Particle(new_state[0], new_state[1], new_state[2],
                                concerned_particle.index)
        new_energy = new_particle.E
        old_energy = concerned_particle.E
        delta_E = new_energy - old_energy

        # Metropolis acceptance criterion (using physical energy)
        coin = np.random.random()
        to_accept = delta_E <= 0 or coin < np.exp(-delta_E / (KB * T))

        if debug:
            print(f'delta_E = {delta_E}; exp = {np.exp(-delta_E / (KB * T))}; coin={coin}')
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
        """Get energy distribution E vs n dictionary"""
        return self.energy_map

    def print_stats(self):
        print("QUANTUM NUMBERS\tOCCUPATION NUMBER")
        print("____________________________________________")
        for k in self.particles_hashmap:
            print(f"{k}\t{len(self.particles_hashmap[k])}")
        print("____________________________________________")


class Ensemble():
    def __init__(self, N, T):
        self.N = N
        self.T = T
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
                        # Compute initial occupation per state
                        self.occupation_history = {E: n / self.system.degeneracy.get(E, 1) 
                                                   for E, n in self.system.energy_map.items()}
                        return
        # If N > total states, still compute
        self.occupation_history = {E: n / self.system.degeneracy.get(E, 1) 
                                   for E, n in self.system.energy_map.items()}

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

        self.occupation_history = average_occupation_per_energy
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

    def be_occupancy_per_state(self, E: float, mu: float, T: float) -> float:
        """Bose–Einstein occupancy for a single quantum state (no spin degeneracy)."""
        x = (E - mu) / (KB * T)
        if x > 700:  # avoid overflow -> occupancy ~ 0
            return 0.0
        if x <= 1e-14:  # extremely close to divergence; cap large
            return 1e14
        return 1.0 / (np.exp(x) - 1.0)

    def find_mu_be(self, energies_sorted, N_target):
        """
        Find chemical potential mu such that:
            N_target = sum_E g(E) * 1/(exp((E - mu)/kBT) - 1)
        using bisection with guaranteed bracketing below E_min.

        NOTE: This is the correct method for bosons (refactored into Ensemble class).
        """
        E_min = min(energies_sorted)
        # Bracket: mu must be < E_min
        mu_low = E_min - 100.0 * KB * self.T  # far below -> small N
        mu_high = E_min - 1e-30          # just below ground -> huge N

        def N_of_mu(mu):
            total = 0.0
            for E in energies_sorted:
                g = self.system.degeneracy.get(E, 1)
                occ = self.be_occupancy_per_state(E, mu, self.T)
                total += g * occ
                if total > 10 * N_target:  # early break to keep numeric stability
                    break
            return total

        N_low = N_of_mu(mu_low)
        N_high = N_of_mu(mu_high)

        # Ensure bracketing (monotone increasing in mu)
        if N_low > N_target:
            # widen lower bound
            for _ in range(50):
                mu_low -= 100.0 * KB * self.T
                N_low = N_of_mu(mu_low)
                if N_low <= N_target:
                    break
        if N_high < N_target:
            # move even closer to E_min from below
            for k in range(50):
                mu_high = E_min - 10.0**(-(k+3))  # E_min - 1e-3, 1e-4, ...
                N_high = N_of_mu(mu_high)
                if N_high >= N_target:
                    break

        # Bisection
        for _ in range(200):
            mu_mid = 0.5 * (mu_low + mu_high)
            N_mid = N_of_mu(mu_mid)
            if abs(N_mid - N_target) <= 1e-6 * max(1.0, N_target):
                return mu_mid
            if N_mid < N_target:
                mu_low = mu_mid
            else:
                mu_high = mu_mid
        return 0.5 * (mu_low + mu_high)

    def plot_stats(self, title):
        """
        Plot simulation statistics and compare with Bose-Einstein distribution.
        Uses the correct chemical potential finding method (refactored from main()).
        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

        # Left plot: Instantaneous particle distribution
        labels = list(self.system.energy_map.keys())
        values = list(self.system.energy_map.values())
        ax[0].scatter(labels, values)
        ax[0].set_xlabel('Energy (J)')
        ax[0].set_ylabel('Number of particles')
        ax[0].set_title('Particle Distribution')
        fig.suptitle(f'Boson Statistics ({title}) N={self.N} T={self.T}K')

        # Right plot: Average occupation per state vs Bose-Einstein prediction
        energy = sorted(self.occupation_history.keys())  # Sort energies
        avg_per_energy = [self.occupation_history[e] for e in energy]  # Average occupation per state

        ax[1].scatter(energy, avg_per_energy, label='Simulation', s=50, alpha=0.7)
        ax[1].set_xlabel('Energy (J)')
        ax[1].set_ylabel('Average Occupation per Quantum State')
        ax[1].set_title('Energy Distribution')

        # Comparison with Bose-Einstein statistics
        # Find μ so that N_theory(μ) = N using the correct method
        degeneracy_map = self.system.degeneracy
        mu = self.find_mu_be(energy, self.N)
        print(f'Computed chemical potential μ = {mu:.6e} J')

        # Calculate Bose-Einstein occupancy for each energy
        be_E = []
        be_f_of_E = []
        for e_val in energy:
            be_E.append(e_val)
            f_of_E = self.be_occupancy_per_state(e_val, mu, self.T)
            be_f_of_E.append(f_of_E)
        max_be = max(be_f_of_E)

        # Check if we have valid BE values and plot
        if max_be > 0:
            # Plot without normalization - use actual values for proper comparison
            #be_f_of_E = np.array(be_f_of_E)
            #be_f_of_E /= max_be
            ax[1].plot(be_E, be_f_of_E, label=f'Bose-Einstein (μ = {mu: .2e})',
                      color='r', linewidth=2, linestyle='--')
            ax[1].legend()
        else:
            print("\nWARNING: Bose-Einstein distribution too small to plot.")
            print(f"Temperature T={self.T}K may be too low for this energy scale.")
            ax[1].legend()

        plt.show()


def main():
    ens = Ensemble(N, T)
    ens.build_ensemble()
    print(f"\nTotal energy (J): {ens.system.E_total}")
    print(f"Average energy per particle (J): {ens.system.E_total/N:.6e}")
    ens.plot_stats('Initial')
    print()

    # Equilibration
    print("Equilibriating...")
    ens.walk(steps_to_equilibrium, record_interval=100)
    ens.plot_stats('Equilibrium')
    print()

    # Production run with recording
    print("Exploring the gamma space...")
    ens.walk(steps_to_explore, record_interval=100)
    print(f"\nFinal total energy (J): {ens.system.E_total}")
    print(f"Average energy per particle (J): {ens.system.E_total/N:.6e}")
    ens.plot_stats("Final")
    print()


if __name__ == '__main__':
    main()
