import numpy as np
import matplotlib.pyplot as plt

# Physical constants
KB = 1.3806452E-23
h = 6.636e-34
hbar = h / (2 * np.pi)

# Simulation parameters
N = 2000
T = 3000
mass = 9.1e-31  # Electron mass
L = 5e-9         # 5 nm
MAX_QUANTUM_NUMBER = 13
debug = False

# round off margin
eps = 3*h**2/(8*mass*L**2)
n_places = np.floor(abs(np.log10(eps)))


class Particle():
    def __init__(self, l, m, n, index):
        self.k = (l, m, n)
        self.E = (h ** 2) * (self.k[0] ** 2 + self.k[1] ** 2 + self.k[2] ** 2) / (8 * mass * L ** 2)
        self.E = round(self.E, n_places-1)
        self.index = index

    def update_energy(self):
        self.E = (h ** 2) * (self.k[0] ** 2 + self.k[1] ** 2 + self.k[2] ** 2) / (8 * mass * L ** 2)


class Microstate():
    def __init__(self, N):
        self.particles_hashmap = {}
        self.particles = {}
        self.energy_map = {}
        self.E = 0

    def add_particle(self, state: Particle):
        key = round(state.E, 25)
        if state.k in self.particles_hashmap:
            if len(self.particles_hashmap[state.k]) == 2:
                return -1
            else:
                self.particles_hashmap[state.k].append(state.index)
                self.particles[state.index] = state
                self.energy_map[key] = self.energy_map.get(key, 0) + 1
                self.E += state.E
                return 0
        else:
            self.particles_hashmap[state.k] = [state.index]
            self.particles[state.index] = state
            self.energy_map[key] = self.energy_map.get(key, 0) + 1
            self.E += state.E
            return 0

    def transition(self, particle_index: int, new_state: tuple):
        if new_state in self.particles_hashmap:
            occupation_number = len(self.particles_hashmap[new_state])
            if occupation_number == 2:
                if debug:
                    print(f"{new_state} state already has {occupation_number} particles!")
                return -1

        concerned_particle = self.particles[particle_index]
        old_state = concerned_particle.k
        new_particle = Particle(new_state[0], new_state[1], new_state[2], concerned_particle.index)
        new_energy = new_particle.E
        old_energy = concerned_particle.E
        delta_E = new_energy - old_energy
        coin = np.random.random()
        to_accept = delta_E <= 0 or coin < np.exp(-delta_E / (KB * T))

        if to_accept:
            # remove from old state
            self.particles_hashmap[old_state].remove(concerned_particle.index)
            if len(self.particles_hashmap[old_state]) == 0:
                del self.particles_hashmap[old_state]

            # energy bookkeeping
            old_key = round(old_energy, 25)
            self.energy_map[old_key] -= 1
            if self.energy_map[old_key] == 0:
                del self.energy_map[old_key]
            self.E -= old_energy

            # update to new state
            concerned_particle.k = new_state
            concerned_particle.update_energy()
            new_key = round(concerned_particle.E, 25)
            self.energy_map[new_key] = self.energy_map.get(new_key, 0) + 1
            self.particles_hashmap.setdefault(new_state, []).append(concerned_particle.index)
            self.E += concerned_particle.E
            return 1
        else:
            return 0


class Ensemble():
    def __init__(self, N):
        self.N = N
        self.system = Microstate(N)
        self.stats = None

    def build_ensemble(self):
        id = 0
        for l in range(1, MAX_QUANTUM_NUMBER + 1):
            for m in range(1, MAX_QUANTUM_NUMBER + 1):
                for n in range(1, MAX_QUANTUM_NUMBER + 1):
                    up_particle = Particle(l, m, n, id)
                    id += 1
                    down_particle = Particle(l, m, n, id)
                    id += 1
                    self.system.add_particle(up_particle)
                    self.system.add_particle(down_particle)
                    if id >= N:
                        return

    def walk(self, n_trials, equilibration_steps=50000, record_interval=200):
        accepted, rejected, forbidden = 0, 0, 0
        energy_records = {}

        for i in range(n_trials):
            chosen_one = np.random.randint(0, N)
            new_l = np.random.randint(1, MAX_QUANTUM_NUMBER + 1)
            new_m = np.random.randint(1, MAX_QUANTUM_NUMBER + 1)
            new_n = np.random.randint(1, MAX_QUANTUM_NUMBER + 1)
            new_quantum_numbers = (new_l, new_m, new_n)

            result = self.system.transition(chosen_one, new_quantum_numbers)
            if result == 1:
                accepted += 1
            elif result == 0:
                rejected += 1
            else:
                forbidden += 1

            # start recording only *after* equilibration
            if i > equilibration_steps and i % record_interval == 0:
                for E, n_occ in self.system.energy_map.items():
                    energy_records[E] = energy_records.get(E, 0) + n_occ

        total_samples = max(1, (n_trials - equilibration_steps) // record_interval)
        averaged = {E: n / total_samples for E, n in energy_records.items()}

        self.stats = {
            "n_trials": n_trials,
            "equilibration_steps": equilibration_steps,
            "accepted": accepted,
            "rejected": rejected,
            "forbidden": forbidden,
            "acceptance_rate": round(accepted / n_trials, 2),
            "rejectance_rate": round(rejected / n_trials, 2),
            "forbiddance_rate": round(forbidden / n_trials, 2)
        }

        print("\n--- Simulation Summary ---")
        for k, v in self.stats.items():
            print(f"{k:20s}: {v}")

        return averaged


def plot_stats(stats):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    text = "\n".join([f"{k:20s}: {v}" for k, v in stats.items()])
    ax.text(0.05, 0.5, text, fontsize=12, family="monospace", va="center")
    plt.title("Monte Carlo Simulation Statistics", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


def main():
    ens = Ensemble(N)
    ens.build_ensemble()
    print(f"Initial ensemble energy: {ens.system.E}")

    # compute degeneracy map explicitly (counts of states per energy)
    degeneracy = {}
    for l in range(1, MAX_QUANTUM_NUMBER + 1):
        for m in range(1, MAX_QUANTUM_NUMBER + 1):
            for n in range(1, MAX_QUANTUM_NUMBER + 1):
                E = (h ** 2) * (l ** 2 + m ** 2 + n ** 2) / (8 * mass * L ** 2)
                key = round(E, 25)
                degeneracy[key] = degeneracy.get(key, 0) + 1

    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel('Energy')
    plt.ylabel('Number of particles')
    plt.title('Fermion Statistics (Initial)')
    plt.show()

    avg_occ = ens.walk(150000, equilibration_steps=50000, record_interval=200)
    print(f"Final ensemble energy: {ens.system.E}")
    print(f"1.5 N.Kb.T = {1.5 * N * KB * T}")

    sorted_E = sorted(avg_occ.keys())
    avg_per_state = [avg_occ[E] / degeneracy.get(E, 1) for E in sorted_E]

    plt.plot(sorted_E, avg_per_state, 'o-', label='Averaged Occupation per State')
    plt.xlabel('Energy')
    plt.ylabel('Average Occupation per Quantum State')
    plt.title('Fermion Energy Distribution (Equilibrium)')
    plt.legend()
    plt.show()

    Ef = sorted_E[len(sorted_E)//3]  # crude estimate for Fermi energy
    fermi_dirac = [1 / (np.exp((E - Ef) / (KB * T)) + 1) for E in sorted_E]
    plt.plot(sorted_E, fermi_dirac, 'r-', label='Fermi–Dirac Distribution (Theory)')
    plt.xlabel('Energy')
    plt.ylabel('Occupation Probability')
    plt.title('Fermi–Dirac Distribution')
    plt.legend()
    plt.show()

    plot_stats(ens.stats)


if __name__ == '__main__':
    main()
