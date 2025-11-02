import numpy as np
import matplotlib.pyplot as plt

# Physical constants
KB = 1.3806452E-23
h = 6.636e-34
hbar = h/(2*np.pi)

# Simulation parameters
N = 200
T = 300
mass = 9.1e-31 # Higgs Boson ... ?
L = 5e-9     # 5 nm
MAX_QUANTUM_NUMBER = 10
debug = False

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
        self.E = (h**2)*(self.k[0]**2 + self.k[1]**2 + self.k[2]**2)/(8*mass*L**2)
        self.index = index

    def update_energy(self):
        '''Updates value of energy of particle'''
        self.E = (h**2)*(self.k[0]**2 + self.k[1]**2 + self.k[2]**2)/(8*mass*L**2)

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

    def transition(self, particle_index:int, new_state:tuple):
        '''
        Execute a particle swap. This changes the energy map of the system
        Returns 
        1  if swap accepted
        0  if swap rejected
        -1 if swap forbidden
        '''
        # check if a particle with given k exists
        if new_state in self.particles_hashmap:
            # is it paired? then swap is forbidden
            occupation_number = len(self.particles_hashmap[new_state])
            if occupation_number == 2:
                if debug:
                    print(f"{new_state} state already has {occupation_number} particles!")
                return -1
            
        # pop a particle with the given k
        concerned_particle = self.particles[particle_index]

        # first create a particle
        new_particle = Particle(new_state[0], new_state[1], new_state[2],
                                concerned_particle.index)
        new_energy = new_particle.E
        old_energy = concerned_particle.E
        delta_E = new_energy-old_energy
        coin = np.random.random() 
        to_accept = delta_E <= 0 or coin < np.exp(-delta_E/(KB*T))
        if debug:
            print(f'delta_E = {delta_E}; exp = {np.exp(-delta_E/(KB*T))}; coin={coin}')
            print(f'{"Accep" if to_accept else "Rejec"}ting jump')

        if to_accept:
            # update particle state and system energy
            self.energy_map[concerned_particle.E] -= 1
            self.E -= concerned_particle.E

            concerned_particle.k = new_state
            concerned_particle.update_energy()
            # make the swap
            self.particles_hashmap[new_state] = [concerned_particle]
            # update energy map
            if concerned_particle.E in self.energy_map:
                self.energy_map[concerned_particle.E] += 1
            else:
                self.energy_map[concerned_particle.E] = 1
            self.E += concerned_particle.E
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

    def build_ensemble(self):
        '''Builds a system of N fermions'''
        id = 0
        for l in range(MAX_QUANTUM_NUMBER):
            for m in range(MAX_QUANTUM_NUMBER):
                for n in range(MAX_QUANTUM_NUMBER):
                    up_particle = Particle(l, m, n, id)
                    id += 1
                    down_particle = Particle(l, m, n, id)
                    id += 1
                    self.system.add_particle(up_particle)
                    self.system.add_particle(down_particle)
                    if id >= N:
                        return

    def walk(self, n_trials):
        '''Take `n_steps` random walks in the gamma space'''
        accepted, rejected, forbidden = 0, 0, 0
        for i in range(n_trials):
            chosen_one = np.random.randint(0, N)
            new_l = np.random.randint(0, MAX_QUANTUM_NUMBER)
            new_m = np.random.randint(0, MAX_QUANTUM_NUMBER)
            new_n = np.random.randint(0, MAX_QUANTUM_NUMBER)
            new_quantum_numbers = (new_l, new_m, new_n)
            if debug:
                print(f'[{i}] : l = {new_l}\tm = {new_m}\tn = {new_n}')
            result = self.system.transition(chosen_one, new_quantum_numbers)
            if result == 1:
                accepted += 1
            elif result == 0:
                rejected += 1
            else:
                forbidden += 1
        acceptance = accepted/n_trials
        rejectance = rejected/n_trials
        forbiddance = forbidden/n_trials
        print(f"Took {n_trials} walks")
        print(f"Accepted : {accepted} ({acceptance*100:0.2f}%)")
        print(f"Rejected : {rejected} ({rejectance*100:0.2f}%)")
        print(f"Forbidden : {forbidden} ({forbiddance*100:0.2f}%)")


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
    ens.walk(5000)
    print(f"Ensemble energy: {ens.system.E}")
    print(f"1.5 N.Kb.T = {1.5*N*KB*T}")
    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel('Energy')
    plt.ylabel('Number of particles')
    plt.title('Fermion Statistics (Equilibrium, 6000 walks)')
    plt.show()

    ens.walk(40000)
    labels = list(ens.system.energy_map.keys())
    values = list(ens.system.energy_map.values())
    plt.scatter(labels, values)
    plt.xlabel('Energy')
    plt.ylabel('Number of particles')
    plt.title('Fermion Statistics (Final, 40000 walks)')
    plt.show()

if __name__ == '__main__':
    main()