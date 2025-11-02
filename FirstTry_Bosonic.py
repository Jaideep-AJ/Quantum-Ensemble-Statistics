import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm

# Physical constants
KB = 1.3806452E-23
h = 6.636e-34
hbar = h/(2*np.pi)

# Simulation parameters
N = 2000
T = 900
mass = 9.1e-31  # Electron mass
L = 5e-9  # 5 nm
MAX_QUANTUM_NUMBER = 13
debug = False

# Energy normalization factor
E0 = h**2/(8*mass*L**2)

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
    """
    def __init__(self, l, m, n, index):
        self.k = (l, m, n)
        self.E_tilde = 0  # Normalized energy
        self.E = 0  # Actual energy
        self.update_energy()
        self.index = index

    def update_energy(self):
        """Updates value of energy of particle"""
        # Normalized energy: E_tilde = l^2 + m^2 + n^2
        self.E_tilde = self.k[0]**2 + self.k[1]**2 + self.k[2]**2
        # Actual energy
        self.E = E0 * self.E_tilde

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
        self.E = 0  # Total actual energy
        self.E_tilde_total = 0  # Total normalized energy

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
            self.energy_map[state.E_tilde] += 1
        else:
            # Create new entry for this quantum state
            self.particles_hashmap[state.k] = [state.index]
            self.particles[state.index] = state
            if state.E_tilde in self.energy_map:
                self.energy_map[state.E_tilde] += 1
            else:
                self.energy_map[state.E_tilde] = 1

        # Update microstate energy
        self.E += state.E
        self.E_tilde_total += state.E_tilde
        return 0

    def transition(self, particle_index: int, new_state: tuple):
        """
        Execute a particle swap. This changes the energy map of the system
        Returns
        1 if swap accepted
        0 if swap rejected

        NOTE: For bosons, NO transitions are forbidden due to Pauli exclusion!
        """

        # Get the particle to transition
        concerned_particle = self.particles[particle_index]

        # Calculate new energy
        new_particle = Particle(new_state[0], new_state[1], new_state[2],
                               concerned_particle.index)
        new_energy = new_particle.E
        old_energy = concerned_particle.E
        delta_E = new_energy - old_energy

        # Metropolis acceptance criterion
        coin = np.random.random()
        to_accept = delta_E <= 0 or coin < np.exp(-delta_E/(KB*T))

        if debug:
            print(f'delta_E = {delta_E}; exp = {np.exp(-delta_E/(KB*T))}; coin={coin}')
            print(f'{"Accep" if to_accept else "Rejec"}ting jump')

        if to_accept:
            old_k = concerned_particle.k
            old_E_tilde = concerned_particle.E_tilde

            # Update energy maps
            self.energy_map[old_E_tilde] -= 1
            if self.energy_map[old_E_tilde] == 0:
                del self.energy_map[old_E_tilde]
            self.E -= old_energy
            self.E_tilde_total -= old_E_tilde

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
            if concerned_particle.E_tilde in self.energy_map:
                self.energy_map[concerned_particle.E_tilde] += 1
            else:
                self.energy_map[concerned_particle.E_tilde] = 1

            self.E += concerned_particle.E
            self.E_tilde_total += concerned_particle.E_tilde
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

        for i in tqdm.tqdm(range(n_trials)):
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

def main():
    ens = Ensemble(N)
    ens.build_ensemble()

    print(f"Ensemble energy: {ens.system.E}")
    print(f"Normalized total energy: {ens.system.E_tilde_total}")

    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel('Normalized Energy (l² + m² + n²)')
    plt.ylabel('Number of particles')
    plt.title('Boson Statistics (Initial)')
    plt.show()

    # Equilibration
    print("Equilibriating...")
    ens.walk(100000, record_interval=-1)
    print(f"Ensemble energy after equilibration: {ens.system.E}")
    print(f"1.5 N.Kb.T = {1.5*N*KB*T}")

    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel('Normalized Energy (l² + m² + n²)')
    plt.ylabel('Number of particles')
    plt.title('Instantaneous Boson Distribution (Equilibrium, 100000 walks)')
    plt.show()

    # Production run with recording
    print("Roaming the gamma space...")
    avg_occ = ens.walk(1000000, record_interval=100)

    print(f"Final ensemble energy: {ens.system.E}")
    print(f"1.5 N.Kb.T = {1.5 * N * KB * T}")

    energy = list(avg_occ.keys())  # normalized energies
    avg_per_energy = list(avg_occ.values())  # average occupation per state

    # Normalize for plotting
    max_avg_occupation = max(avg_per_energy)
    energy_prob = [avg / max_avg_occupation for avg in avg_per_energy]

    plt.scatter(energy, energy_prob, label='Simulation')
    plt.xlabel('Normalized Energy (l² + m² + n²)')
    plt.ylabel('Occupation Probability per Quantum State')
    plt.title('Boson Energy Distribution (Final)')

    # Comparison with Bose-Einstein statistics
    # For chemical potential μ ≈ 0 (non-degenerate case)
    sorted_energy = sorted(energy)
    be_E = []
    be_f_of_E = []

    for e_tilde in sorted_energy:
        e_actual = E0 * e_tilde
        be_E.append(e_tilde)
        # Bose-Einstein distribution with μ = 0
        f_of_E = 1 / (np.exp(e_actual / (KB * T)) - 1)
        be_f_of_E.append(f_of_E)

    # Normalize BE distribution for comparison
    max_be = max(be_f_of_E)
    be_f_of_E_normalized = [f / max_be for f in be_f_of_E]

    plt.plot(be_E, be_f_of_E_normalized, label='Bose-Einstein Statistics (μ=0)', 
             color='r', linewidth=2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
