import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Physical constants
KB = 1.3806452E-23
h = 6.636e-34
hbar = h/(2*np.pi)

# Simulation parameters
N = 2000
T = 3000
steps_to_equilibrium = 50000
steps_to_explore = 150000
mass = 9.1e-31 # Higgs Boson ... ?
L = 5e-9     # 5 nm
MAX_QUANTUM_NUMBER = 13
debug = False

# round off margin
eps = 3*h**2/(8*mass*L**2)
#n_places = int(np.floor(abs(np.log10(eps))))
#print(n_places)
n_places = 25

# Emax = 1.5*KB*T*N*1
Emax = 1

class Particle():
    '''
    Create a Particle in a 3D box

    ----
    PARAMETERS
    ----
    l : wave vector in x
    m : wave vector in y
    n : wave vector in z
    s : spin of fermion
    index : particle identifier "name" 
    '''

    def __init__(self, l, m, n, index):
        self.k = (l, m, n)
        self.E = 0
        self.update_energy()
        self.index = index

    def update_energy(self):
        '''Updates value of energy of particle'''
        self.E = (h**2)*(self.k[0]**2 + self.k[1]**2 + self.k[2]**2)/(8*mass*L**2)
        self.E = round(self.E, n_places)

    def __eq__(self, other):
        '''Check equivalence between particles'''
        try:
            if self.k[0] == other.k[0] and self.k[1] == other.k[1] and self.k[2] == other.k[2]:
                return True
            else:
                return False
        except Exception:
            return False
        

class Microstate():
    '''
    Create a certain configuration of N particles in a 3D box

    ----
    PARAMETERS
    ----
    N : Number of particles
    '''
    def __init__(self, N):
        self.particles_hashmap = {}
        self.particles = {}
        self.energy_map = {}
        self.E = 0
        self.degeneracy = {}
        for l in range(1, MAX_QUANTUM_NUMBER + 1):
            for m in range(1, MAX_QUANTUM_NUMBER + 1):
                for n in range(1, MAX_QUANTUM_NUMBER + 1):
                    E = (h ** 2) * (l ** 2 + m ** 2 + n ** 2) / (8 * mass * L ** 2)
                    key = round(E, n_places)
                    self.degeneracy[key] = self.degeneracy.get(key, 0) + 1

    def add_particle(self, state:Particle):
        '''Add a particle to the ensemble while building it. 
            Returns 0 is addition successful or -1 if addition forbidden'''
        if state.k in self.particles_hashmap:
            # each energy state can hold two fermions (spins +0.5 and -0.5)
            if len(self.particles_hashmap[state.k]) == 2:
                return -1
            else:
                # add fermion to the state if state is not full
                self.particles_hashmap[state.k].append(state.index)
                self.particles[state.index] = state
                self.energy_map[state.E] += 1
                # update microstate energy
                self.E += state.E
                return 0
        else:
            # add fermion to the state, creating the entries in the hashmap
            self.particles_hashmap[state.k] = [state.index]
            self.particles[state.index] = state
            if state.E in self.energy_map:
                self.energy_map[state.E] += 1
            else:
                self.energy_map[state.E] = 1
            # update microstate energy
            self.E += state.E
            return 0

    def transition(self, particle_index:int, new_state_k:tuple):
            concerned_particle = self.particles[particle_index]
            old_state_k = concerned_particle.k

            # if not (1 <= new_state_k[0] <= MAX_QUANTUM_NUMBER and
            #     1 <= new_state_k[1] <= MAX_QUANTUM_NUMBER and
            #     1 <= new_state_k[2] <= MAX_QUANTUM_NUMBER):
            #     return -1

            if new_state_k in self.particles_hashmap and len(self.particles_hashmap[new_state_k]) == 2:
                if debug:
                    print(f"{new_state_k} state already has 2 particles! Forbidden.")
                return -1

            temp_new_particle = Particle(new_state_k[0], new_state_k[1], new_state_k[2], concerned_particle.index)
            new_energy = temp_new_particle.E
            old_energy = concerned_particle.E
            delta_E = new_energy - old_energy

            coin = np.random.random()
            to_accept = delta_E <= 0 or coin < np.exp(-delta_E / (KB * T))

            if to_accept:
                self.energy_map[old_energy] -= 1
                self.particles_hashmap[old_state_k].remove(particle_index)
                if not self.particles_hashmap[old_state_k]:
                    del self.particles_hashmap[old_state_k]

                concerned_particle.k = new_state_k
                concerned_particle.update_energy()

                if new_state_k not in self.particles_hashmap:
                    self.particles_hashmap[new_state_k] = []
                self.particles_hashmap[new_state_k].append(particle_index)

                self.energy_map[concerned_particle.E] = self.energy_map.get(concerned_particle.E, 0) + 1
                self.E += delta_E
                return 1
            else:
                return 0    

    def get_stats(self):
        '''Get energy distribution E vs n dictionary'''
        return self.energy_map
    
    def print_stats(self):
        print("QUANTUM NUMBERS\tOCCUPATION NUMBER")
        print("____________________________________________")
        for k in self.particles_hashmap:
            print(f"{k}\t{self.particles_hashmap[k]}")
        print("____________________________________________")
        

class Ensemble():
    def __init__(self, N):
        self.N = N
        self.system = Microstate(N)
        self.Ef = 0

    def build_ensemble(self):
        '''Builds a system of N fermions'''
        id = 0
        for l in range(1, MAX_QUANTUM_NUMBER+1):
            for m in range(1, MAX_QUANTUM_NUMBER+1):
                for n in range(1, MAX_QUANTUM_NUMBER+1):
                    up_particle = Particle(l, m, n, id)
                    id += 1
                    down_particle = Particle(l, m, n, id)
                    id += 1
                    self.system.add_particle(up_particle)
                    self.system.add_particle(down_particle)
                    if id >= N:
                        self.Ef = up_particle.E
                        return

    def walk(self, n_trials, record_interval=200):
        '''Take `n_steps` random walks in the gamma space
        `record_interval` must be > 0 for recording stats'''
        accepted, rejected, forbidden = 0, 0, 0
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
            elif result == 0:
                rejected += 1
            else:
                forbidden += 1

            if record_interval > 0 and i % record_interval == 0:
                # for E, n_occ in self.system.energy_map.items():
                #     energy_records[E] = energy_records.get(E, 0) + n_occ
                for energy, n_particles in self.system.energy_map.items():
                    occupation_numbers[energy] += n_particles
                n_samples += 1

        average_occupation_per_energy = {}
        for energy, n_particles in occupation_numbers.items():
            degen = self.system.degeneracy.get(energy, 1)
            average_occupation_per_energy[energy] = n_particles/(n_samples*degen)
                    

        # total_samples = max(1, (n_trials - equilibration_steps) // record_interval)
        # averaged = {E: n / (total_samples*E) if E != 0 else 1 for E, n in energy_records.items()}

        self.stats = {
            "n_trials": n_trials,
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

        return average_occupation_per_energy

    def find_chemical_potential(self):
        # Use the full degeneracy map (all single-particle (l,m,n) levels),
        # and include the spin factor (2).
        energies = np.array(sorted(self.system.degeneracy.keys()))
        degeneracies = np.array([self.system.degeneracy[e] for e in energies], dtype=float)
        degeneracies *= 2.0   # factor 2 for spin

        # Fermi-Dirac expectation for total particle number
        def expectation_N(mu):
            f = 1.0 / (np.exp((energies - mu) / (KB * T)) + 1.0)
            return np.sum(degeneracies * f)

        # sensible bracket (mu increases => expectation_N increases)
        mu_low = np.min(energies) - 10.0 * KB * T
        mu_high = np.max(energies) + 10.0 * KB * T

        # Sanity: if even at mu_high you don't reach N, increase mu_high
        # (or raise an error)
        if expectation_N(mu_high) < self.N:
            raise RuntimeError("Not enough available single-particle states to accommodate N. Increase MAX_QUANTUM_NUMBER.")

        # Bisection (correct direction)
        for _ in range(200):
            mu_mid = 0.5 * (mu_low + mu_high)
            if expectation_N(mu_mid) < self.N:
                mu_low = mu_mid     # increase mu to increase N
            else:
                mu_high = mu_mid
        return 0.5 * (mu_low + mu_high)


def main():
    ens = Ensemble(N)
    ens.build_ensemble()
    print(f"Ensemble energy: {ens.system.E}")
    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel('Energy')
    plt.ylabel('Number of particles')
    plt.title('Fermion Statistics (Initial)')
    plt.show()

    #ens.system.print_stats()
    print("Equilibriating...")
    ens.walk(steps_to_equilibrium, record_interval=-1)
    print()
    print(f"Ensemble energy: {ens.system.E}")
    print(f"1.5 N.Kb.T = {1.5*N*KB*T}")
    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel('Energy')
    plt.ylabel('Number of particles')
    plt.title('Instantaneous Fermion Distribution (Equilibrium, 100000 walks)')
    plt.show()

    print("Exploring the gamma space...")
    avg_occ = ens.walk(steps_to_explore, record_interval=100)
    print()
    print(f"Final ensemble energy: {ens.system.E}")
    print(f"1.5 N.Kb.T = {1.5 * N * KB * T}")

    energy = avg_occ.keys() # energies
    avg_per_energy = avg_occ.values() # energy per state
    max_avg_occupation = max(avg_per_energy)
    energy_prob = [avg / max_avg_occupation for avg in avg_per_energy]
    plt.scatter(energy, energy_prob, label='Simulation', s=50, alpha=0.7)
    plt.xlabel('Energy')
    plt.ylabel('Occupation Probability per Quantum State')
    plt.title('Fermion Energy Distribution (Final)')

    # comparison with fermi_dirac stats
    fd_E = []
    fd_f_of_E = []
    sorted_energy = sorted(energy)
    #Ef = sorted_energy[len(sorted_energy)//2]
    Ef = ens.find_chemical_potential()
    print(f'Ef = {Ef}')
    #Ef = ens.Ef
    #Ef = energy[energy_prob.index(0.5)]
    for e in sorted_energy:
        fd_E.append(e)
        f_of_E = 1 / (np.exp((e - Ef) / (KB * T)) + 1)
        fd_f_of_E.append(f_of_E)
    plt.plot(fd_E, fd_f_of_E, label='Fermi-Dirac Statistics', 
             color='r', linewidth=2, linestyle='--')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    main()