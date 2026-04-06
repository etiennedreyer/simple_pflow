import jax
import jax.numpy as jnp
from jax import random
from functools import partial

class EventGenerator:

    def __init__(self, config):
        self.config = config
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']
        self.power = config.get('power', 2.0)
        self.pad_value = float(config.get('pad_value', 0.0))
        self.seed = config.get('seed', 0)

    def set_seed(self, seed):
        self.seed = seed

    @partial(jax.jit, static_argnames=['self', 'N_events'])
    def generate(self, N_events=1):

        key = random.key(self.seed)
        key, k1, k2, k3, k4 = random.split(key, 5)

        N_particles = random.randint(k1, (N_events,), self.N_min, self.N_max + 1)

        ### Uniform spread in x/y
        particle_xs = random.uniform(k2, (N_events, self.N_max)) * (self.x_max - self.x_min) + self.x_min
        particle_ys = random.uniform(k3, (N_events, self.N_max)) * (self.y_max - self.y_min) + self.y_min

        ### Power-law spectrum in energy
        r = random.uniform(k4, (N_events, self.N_max))
        particle_Es = ((self.E_max**(1-self.power) - self.E_min**(1-self.power)) * r + self.E_min**(1-self.power))**(1/(1-self.power))

        if N_events > 1:
            ### Assign pad_value to padded particles
            index = jnp.arange(self.N_max)[None, :]  # (1, self.N_max)
            mask = index >= N_particles[:, None]  # (N_events, self.N_max)
            particle_Es = jnp.where(mask, self.pad_value, particle_Es)
            particle_xs = jnp.where(mask, self.pad_value, particle_xs)
            particle_ys = jnp.where(mask, self.pad_value, particle_ys)

        return particle_Es, particle_xs, particle_ys