from functools import partial

import jax
import jax.numpy as jnp
from .calo_flash import shoot

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

        self.initialize_cells()
    
    def initialize_cells(self):
        self.cell_size_x = self.width / self.N_cells_x
        self.cell_size_y = self.height / self.N_cells_y
        self.cell_size_z = self.depth / self.N_cells_z

        # N+1 bin edges covering the full range
        self.cell_x_edges = jnp.linspace(-self.width/2, self.width/2, self.N_cells_x + 1)
        self.cell_y_edges = jnp.linspace(-self.height/2, self.height/2, self.N_cells_y + 1)
        self.cell_z_edges = jnp.linspace(0, self.depth, self.N_cells_z + 1)

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

        ### Mask out padded (E == 'nan') particles
        real_part_idx = jnp.where(flat_Es == flat_Es)[0]

        ### Calo Flash simulation (only valid particles)
        ### output values have shape (len(real_part_idx) * N_cells_z * N_spots_per_layer,)
        spot_dict = shoot(flat_Es[real_part_idx], self.Z, 
                      self.cell_z_edges, N_spots_per_layer=N_spots_per_layer)

        ### Map particle idx on real particles back to global particle idx
        spot_global_part_idx  = real_part_idx[spot_dict['particle_idx']]
        spot_event_idx        = spot_global_part_idx // N_particles
        spot_local_part_idx   = spot_global_part_idx  % N_particles

        ### Convert to Cartesian
        spot_x = spot_dict['r'] * jnp.cos(spot_dict['phi']) + flat_xs[spot_global_part_idx]
        spot_y = spot_dict['r'] * jnp.sin(spot_dict['phi']) + flat_ys[spot_global_part_idx]
        spot_z = spot_dict['t']
        spot_e = spot_dict['E']

        ### Discard out-of-bounds spots
        mask = (spot_x >= -self.width/2)  & (spot_x < self.width/2) & \
               (spot_y >= -self.height/2) & (spot_y < self.height/2) & \
               (spot_z >= 0)              & (spot_z < self.depth)
        spot_x, spot_y, spot_z, spot_e = spot_x[mask], spot_y[mask], spot_z[mask], spot_e[mask]
        spot_event_idx, spot_local_part_idx = spot_event_idx[mask], spot_local_part_idx[mask]

        ### Find the local cell index on (x,y,z) axes where the spot falls (floor-division on uniform grid)
        spot_local_cell_idx_x = ((spot_x - self.cell_x_edges[0]) / self.cell_size_x).astype(jnp.int32).clip(0, self.N_cells_x - 1)
        spot_local_cell_idx_y = ((spot_y - self.cell_y_edges[0]) / self.cell_size_y).astype(jnp.int32).clip(0, self.N_cells_y - 1)
        spot_local_cell_idx_z = ((spot_z - self.cell_z_edges[0]) / self.cell_size_z).astype(jnp.int32).clip(0, self.N_cells_z - 1)

        ### Combine these to get a single local cell index running over all N_cells in each event
        spot_local_cell_idx = spot_local_cell_idx_x * (self.N_cells_y * self.N_cells_z) \
                            + spot_local_cell_idx_y * self.N_cells_z \
                            + spot_local_cell_idx_z

        ### Now map to a global cell index running over the entire batch (N_events * N_cells)
        spot_global_cell_idx = spot_event_idx * self.N_cells + spot_local_cell_idx

        ### Aggregate energy into flat (N_events * N_cells) array
        cell_e = jnp.zeros(N_events * self.N_cells)
        cell_e = cell_e.at[spot_global_cell_idx].add(spot_e)

        ### Add noise
        ### (TODO)

        ### Zero out cells below detection threshold
        if self.cell_e_threshold > 0:
            cell_e = cell_e.at[cell_e < self.cell_e_threshold].set(0.0)

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

        ### Hit (active cell) global index running over (N_events * N_cells)
        hit_global_cell_idx = jnp.where(cell_e > 0)[0]

        ### Tally hits per event
        hit_event_idx  = hit_global_cell_idx // self.N_cells
        event_num_hits = jnp.bincount(hit_event_idx, minlength=N_events)
        event_hit_offset = event_num_hits.cumsum(0) - event_num_hits

        hit_dict = {}
        if return_hits:

            ### Hit local index within each event
            hit_global_idx   = jnp.arange(len(hit_global_cell_idx))
            hit_local_idx    = hit_global_idx - event_hit_offset[hit_event_idx]

            ### Hit local cell index
            hit_local_cell_idx = hit_global_cell_idx  % self.N_cells

            ### TODO: can the mapping of index -> xyz be precomputed?
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
                'hit_event_idx': hit_event_idx,
                'hit_idx': hit_local_idx,
                'hit_cell_idx': hit_local_cell_idx,
                'hit_x': hit_x,
                'hit_y': hit_y,
                'hit_z': hit_z,
                'hit_e': hit_e
            }

        truth_dict = {}
        if return_truth:

            ### Compute index which tells which truth (particle, cell) pair to associate with each spot
            spot_global_truth_idx = spot_global_cell_idx * N_particles + spot_local_part_idx

            ### Aggregate spots into flat (N_events * N_cells * N_particles) array of truth energy
            truth_e = jnp.zeros(N_events * self.N_cells * N_particles)
            truth_e = truth_e.at[spot_global_truth_idx].add(spot_e)
            truth_global_idx = jnp.where(truth_e > 0)[0]

            ### Compute local (event, cell, particle) indices corresponding to each nonzero truth entry
            truth_event_idx       = (truth_global_idx // N_particles) // self.N_cells
            truth_local_cell_idx  = (truth_global_idx // N_particles)  % self.N_cells
            truth_local_part_idx  =  truth_global_idx                  % N_particles
            truth_e               = truth_e[truth_global_idx]

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
                truth_hit_is_valid   = truth_global_hit_idx >= 0
                truth_event_idx      = truth_event_idx[truth_hit_is_valid]
                truth_local_cell_idx = truth_local_cell_idx[truth_hit_is_valid]
                truth_local_hit_idx  = truth_local_hit_idx[truth_hit_is_valid]
                truth_local_part_idx = truth_local_part_idx[truth_hit_is_valid]
                truth_e              = truth_e[truth_hit_is_valid]

            truth_dict = {
                'truth_event_idx':    truth_event_idx,
                'truth_cell_idx':     truth_local_cell_idx,
                'truth_hit_idx':      truth_local_hit_idx,
                'truth_particle_idx': truth_local_part_idx,
                'truth_e':            truth_e
            }

        return {**grid_dict, 
                **hit_dict, 
                **truth_dict}