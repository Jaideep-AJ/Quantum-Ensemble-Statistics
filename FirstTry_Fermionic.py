import numpy as np

# Physical constants
KB = 1.3806452E-23
h = 6.636e-34
hbar = h/(2*np.pi)

# Simulation parameters
N = 100
T = 300
m = 1.99e-25 # Higgs Boson
L = 5e-9     # 5 nm

# Emax = 1.5*KB*T*N*1
Emax = 1

class Particle():
    def __init__(self, l, m, n, index):
        self.k = (l, m, n)
        self.E = (h**2)*(self.l**2 + self.m**2 + self.n**2)/(8*m*L**2)
        self.index = index

    def update_energy(self):
        self.E = (h**2)*(self.l**2 + self.m**2 + self.n**2)/(8*m*L**2)

    def __eq__(self, other):
        try:
            if self.l == other.l and self.m == other.m and self.n == other.n:
                return True
            else:
                return False
        except Exception:
            return False
        

class Microstate():
    def __init__(self, E, N, particles):
        self.E = E
        self.particles_hashmap = {}
        self.particles = {}
        self.energy_map = {}

    def add_particle(self, state:Particle):
        if state.k in self.particles_hashmap:
            return -1
        else:
            self.particles_hashmap[state.k] = state.index
            self.particles[state.index] = state
            if state.E in self.energy_map:
                self.energy_map[state.E] += 1
            else:
                self.energy_map[state.E] = 1

    def transition(self, particle_index:int, new_state:tuple):
        if new_state in self.particles_hashmap:
            return -1
        else:
            concerned_particle = self.particles[particle_index]
            del self.particles_hashmap[concerned_particle.k]
            self.energy_map[concerned_particle.E] -= 1
            concerned_particle.k = new_state
            concerned_particle.update_energy()
            self.particles_hashmap[new_state] = concerned_particle
            if concerned_particle.E in self.energy_map:
                self.energy_map[concerned_particle.E] += 1
            else:
                self.energy_map[concerned_particle.E] = 1

    def get_stats(self):
        return self.energy_map
        

class Ensemble():
    def __init__(self, N, ):
        pass