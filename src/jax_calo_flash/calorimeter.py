from functools import partial

import jax
import jax.numpy as jnp
from .calo_flash import shoot
from functools import partial

class CaloBlock:

    ### Homogeneous calorimeter block with cells on Cartesian grid

    def __init__(self, config):
        self.config = config
        self.Z = config['Z']
        self.width = config['width']
        self.height = config.get('height', self.width)
        self.depth = config['depth']
        self.N_cells_x = config['N_cells_x']
        self.N_cells_y = config.get('N_cells_y', self.N_cells_x)
        self.N_cells_z = config['N_cells_z']
        self.N_cells = self.N_cells_x * self.N_cells_y * self.N_cells_z
        self.N_spots_per_layer = config.get('N_spots_per_layer', 1000)
        self.cell_e_threshold = config.get('cell_e_threshold', 0.0)
        self.seed = config.get('seed', 0)

        self.initialize_cells()
    
    def initialize_cells(self):
        self.cell_size_x = self.width / self.N_cells_x
        self.cell_size_y = self.height / self.N_cells_y
        self.cell_size_z = self.depth / self.N_cells_z

        # N+1 bin edges covering the full range
        self.cell_x_edges = jnp.linspace(-self.width/2, self.width/2, self.N_cells_x + 1)
        self.cell_y_edges = jnp.linspace(-self.height/2, self.height/2, self.N_cells_y + 1)
        self.cell_z_edges = jnp.linspace(0, self.depth, self.N_cells_z + 1)

    def set_seed(self, seed):
        self.seed = seed

    @partial(jax.jit, static_argnames=['self', 'return_grid', 'return_hits', 'return_truth', 'N_spots_per_layer'])
    def simulate(self, particle_Es: jax.Array, particle_xs: jax.Array, particle_ys: jax.Array,
                 return_grid=True, return_hits=True, return_truth=True, N_spots_per_layer=None):

        '''
        Six separate sets are involved, each with a different number of elements:
            - events (event)       : separate simulations each containing multiple particles. Serves as the batch dimension.
            - particles (part)     : padded to a fixed maximum (N_particles) across the batch, with E=0 for padding entries
            - spots (spot)         : sampled energy deposits from calo_flash, set by N_spots_per_layer
            - cells (cell)         : the physical volume segments of the calorimeter, whether empty or filled
            - hits (hit)           : subset of cells that have nonzero energy deposits after thresholding
            - truth record (truth) : (src, dst, wgt) triplets connecting hits to parent particles

        Batch computations are done using flattened arrays, for example:
            - input particle properties:  (N_events * N_particles,)
            - simulated cell energies:    (N_events * N_cells,)
            - truth cell-particle record: (N_events * N_cells * N_particles,)

        Indices are labeled as follows:
            - "global" indexes the entire flattened batch
            - "local"  indexes within each event
        '''
        if N_spots_per_layer is None:
            N_spots_per_layer = self.N_spots_per_layer

        ### Auto-unsqueeze (N_particles,) -> (1, N_particles)
        squeezed = particle_Es.ndim == 1
        if squeezed:
            particle_Es = particle_Es[None, :]
            particle_xs = particle_xs[None, :]
            particle_ys = particle_ys[None, :]

        ### Shapes (Note: N_particles = maximum across batch)
        N_events, N_particles = particle_Es.shape

        ### Flatten: (N_events, N_particles) -> (N_events * N_particles,)
        flat_Es = particle_Es.reshape(-1)
        flat_xs = particle_xs.reshape(-1)
        flat_ys = particle_ys.reshape(-1)

        ### Calo Flash simulation
        ### output values have shape (N_particles * N_cells_z * N_spots_per_layer,)
        spot_dict = shoot(flat_Es, self.Z, self.cell_z_edges, 
                          seed=self.seed, N_spots_per_layer=N_spots_per_layer)

        ### Convert to Cartesian, adding per-particle offsets
        spot_x = spot_dict['r'] * jnp.cos(spot_dict['phi']) + flat_xs[spot_dict['particle_idx']]
        spot_y = spot_dict['r'] * jnp.sin(spot_dict['phi']) + flat_ys[spot_dict['particle_idx']]
        spot_z = spot_dict['t']
        spot_e = spot_dict['E']

        ### Label spots as valid if inside geometry and nonzero (not padded)
        spot_valid = (spot_x >= -self.width/2)  & (spot_x < self.width/2) & \
                     (spot_y >= -self.height/2) & (spot_y < self.height/2) & \
                     (spot_z >= 0)              & (spot_z < self.depth) & \
                     (spot_e > 0)
        
        ### Zero out invalid spots in energy
        spot_e = jnp.where(spot_valid, spot_e, 0.0)

        ### Find the local cell index on (x,y,z) axes where the spot falls (floor-division on uniform grid)
        spot_local_cell_idx_x = ((spot_x - self.cell_x_edges[0]) / self.cell_size_x).astype(jnp.int32).clip(0, self.N_cells_x - 1)
        spot_local_cell_idx_y = ((spot_y - self.cell_y_edges[0]) / self.cell_size_y).astype(jnp.int32).clip(0, self.N_cells_y - 1)
        spot_local_cell_idx_z = ((spot_z - self.cell_z_edges[0]) / self.cell_size_z).astype(jnp.int32).clip(0, self.N_cells_z - 1)

        ### Combine these to get a single local cell index running over all N_cells in each event
        spot_local_cell_idx = spot_local_cell_idx_x * (self.N_cells_y * self.N_cells_z) \
                            + spot_local_cell_idx_y * self.N_cells_z \
                            + spot_local_cell_idx_z
        spot_local_cell_idx = jnp.where(spot_valid, spot_local_cell_idx, -1)

        ### Which particle and event associated with each spot
        spot_global_part_idx = jnp.where(spot_valid, spot_dict['particle_idx'], -1)
        spot_event_idx       = jnp.where(spot_valid, spot_global_part_idx // N_particles, -1)
        spot_local_part_idx  = jnp.where(spot_valid, spot_global_part_idx  % N_particles, -1)

        ### Now map to a global cell index running over the entire batch (N_events * N_cells)
        spot_global_cell_idx = spot_event_idx * self.N_cells + spot_local_cell_idx
        spot_global_cell_idx = jnp.where(spot_valid, spot_global_cell_idx, -1)

        ### Aggregate energy into flat (N_events * N_cells) array
        cell_e = jnp.zeros(N_events * self.N_cells)
        cell_e = cell_e.at[spot_global_cell_idx].add(spot_e)

        ### Add noise
        ### (TODO)

        ### Zero out cells below detection threshold
        if self.cell_e_threshold > 0:
            cell_e = jnp.where(cell_e < self.cell_e_threshold, 0.0, cell_e)

        grid_dict = {}
        if return_grid:
            ### Arrange energy deposits into grid (unflatten)
            grid_e = cell_e.reshape(N_events, self.N_cells_x, self.N_cells_y, self.N_cells_z)

            ### remove event dimension if input was squeezed
            if squeezed: grid_e = grid_e[0]

            grid_dict['cell_e'] = grid_e

        ### Fast return
        if not return_hits and not return_truth:
            return grid_dict

        ### Hits = active cells
        hit_valid = cell_e > 0
        hit_global_cell_idx = jnp.arange(len(cell_e))
        hit_event_idx = hit_global_cell_idx // self.N_cells

        ### Tally hits per event
        event_num_hits = hit_valid.reshape(N_events, self.N_cells).sum(axis=1)
        event_hit_offset = event_num_hits.cumsum() - event_num_hits

        hit_dict = {}
        if return_hits:

            ### Hit local index within each event
            hit_local_idx = jnp.arange(len(cell_e)) - event_hit_offset[hit_event_idx]

            ### Hit local cell index
            hit_local_cell_idx = hit_global_cell_idx % self.N_cells

            ### Convert local cell index -> local cell index on (x,y,z) axes
            hit_local_cell_idx_x =  hit_local_cell_idx // (self.N_cells_y * self.N_cells_z)
            hit_local_cell_idx_y = (hit_local_cell_idx  % (self.N_cells_y * self.N_cells_z)) // self.N_cells_z
            hit_local_cell_idx_z =  hit_local_cell_idx  %  self.N_cells_z

            ### Compute corresponding cell centers
            hit_x = self.cell_x_edges[0] + (hit_local_cell_idx_x + 0.5) * self.cell_size_x
            hit_y = self.cell_y_edges[0] + (hit_local_cell_idx_y + 0.5) * self.cell_size_y
            hit_z = self.cell_z_edges[0] + (hit_local_cell_idx_z + 0.5) * self.cell_size_z
            hit_e = cell_e[hit_global_cell_idx]

            hit_dict = {
                'hit_event_idx': jnp.where(hit_valid, hit_event_idx, -1),
                'hit_idx':       jnp.where(hit_valid, hit_local_idx, -1),
                'hit_cell_idx':  jnp.where(hit_valid, hit_local_cell_idx, -1),
                'hit_x':         jnp.where(hit_valid, hit_x, -1),
                'hit_y':         jnp.where(hit_valid, hit_y, -1),
                'hit_z':         jnp.where(hit_valid, hit_z, -1),
                'hit_e':         jnp.where(hit_valid, hit_e, 0.0),
                'hit_valid':     hit_valid
            }

        truth_dict = {}
        if return_truth:

            ### Compute index which tells which truth (particle, cell) pair to associate with each spot
            spot_global_truth_idx = jnp.where(spot_valid,
                                              spot_global_cell_idx * N_particles + spot_local_part_idx, 
                                              -1)

            ### Aggregate spots into flat (N_events * N_cells * N_particles) array of truth energy
            truth_e = jnp.zeros(N_events * self.N_cells * N_particles)
            truth_e = truth_e.at[spot_global_truth_idx].add(spot_e)

            ### Validity
            truth_valid = truth_e > 0

            ### Compute local (event, cell, particle) indices corresponding to each nonzero truth entry
            truth_global_idx      = jnp.arange(len(truth_e))
            truth_event_idx       = (truth_global_idx // N_particles) // self.N_cells
            truth_local_cell_idx  = (truth_global_idx // N_particles)  % self.N_cells
            truth_local_part_idx  =  truth_global_idx                  % N_particles

            ### Compute global hit index for each cell (-1 if inactive/thresholded)
            cell_global_hit_idx = jnp.full((N_events * self.N_cells,), -1, dtype=jnp.int32)
            cell_global_hit_idx = cell_global_hit_idx.at[hit_global_cell_idx].set(jnp.arange(len(hit_global_cell_idx)))

            ### Compute global hit index for each truth entry
            truth_global_cell_idx = truth_event_idx * self.N_cells + truth_local_cell_idx
            truth_global_hit_idx = cell_global_hit_idx[truth_global_cell_idx]

            ### Compute local hit index for each truth entry by subtracting offsets
            truth_local_hit_idx = truth_global_hit_idx - event_hit_offset[truth_event_idx]

            if self.cell_e_threshold > 0:
                ### Drop entries if E>0 but hit failed threshold
                truth_valid = truth_valid & (truth_global_hit_idx >= 0)

            truth_dict = {
                'truth_event_idx':    jnp.where(truth_valid, truth_event_idx, -1),
                'truth_cell_idx':     jnp.where(truth_valid, truth_local_cell_idx, -1),
                'truth_hit_idx':      jnp.where(truth_valid, truth_local_hit_idx, -1),
                'truth_particle_idx': jnp.where(truth_valid, truth_local_part_idx, -1),
                'truth_e':            jnp.where(truth_valid, truth_e, 0.0),
                'truth_valid':        truth_valid
            }

        return {**grid_dict, 
                **hit_dict, 
                **truth_dict}