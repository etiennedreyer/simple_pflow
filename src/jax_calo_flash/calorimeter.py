from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from .calo_flash import shoot
from functools import partial

class CaloBase:

    ### Homogeneous calorimeter simulation with no geometry specified for now

    def __init__(self, config):
        self.config = config
        self.Z = config['Z']
        self.N_spots_per_layer = config.get('N_spots_per_layer', 1000)
        self.cell_e_threshold = config.get('cell_e_threshold', 0.0)
        self.seed = config.get('seed', 0)

        ### Build geometry
        self.geometry = config['geometry']
        self.coords = list(self.geometry.keys())
        self.size = [self.geometry[c]['size'] for c in self.coords]
        self.N_cells = [self.geometry[c]['N_cells'] for c in self.coords]
        self.N_cells_tot = prod(self.N_cells)
        self.cell_sizes = [self.size[i]  / self.N_cells[i] for i in range(3)]

        ### z segmentation is common, but need to implement first two entries in child class
        self.cell_edges = [None, None, jnp.linspace(0, self.size[2], self.N_cells[2] + 1)]

    def set_seed(self, seed):
        self.seed = seed

    def _spots_to_local_cell_idx(self, spots: dict, particle_xs: jax.Array, particle_ys: jax.Array) -> jax.Array:
        pass

    def _local_cell_idx_to_coords(self, local_cell_idx: jax.Array) -> list:
        pass

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

        ### Shapes (Note: N_particles = maximum across batch)
        N_events, N_particles = particle_Es.shape

        ### Flatten: (N_events, N_particles) -> (N_events * N_particles,)
        flat_Es = particle_Es.reshape(-1)

        ### Calo Flash simulation
        ### output values have shape (N_particles * N_cells_z * N_spots_per_layer,)
        spot_dict = shoot(flat_Es, self.Z, self.cell_edges[2], 
                          seed=self.seed, N_spots_per_layer=N_spots_per_layer)

        ### lookup local cell index (geometry-dependent) and flag invalid spots
        spot_local_cell_idx = self._spots_to_local_cell_idx(spot_dict, particle_xs, particle_ys)

        ### Valid spots are contained within geometry and have nonzero energy (i.e. not padded)
        spot_valid = (spot_local_cell_idx >= 0) & (spot_dict['E'] >  0)
        
        ### Which particle and event associated with each spot
        spot_global_part_idx = jnp.where(spot_valid, spot_dict['particle_idx'], -1)
        spot_event_idx       = jnp.where(spot_valid, spot_global_part_idx // N_particles, -1)
        spot_local_part_idx  = jnp.where(spot_valid, spot_global_part_idx  % N_particles, -1)

        ### Now map to a global cell index running over the entire batch (N_events * N_cells)
        spot_global_cell_idx = spot_event_idx * self.N_cells_tot + spot_local_cell_idx
        spot_global_cell_idx = jnp.where(spot_valid, spot_global_cell_idx, -1)

        ### Aggregate energy into flat (N_events * N_cells) array
        spot_e = jnp.where(spot_valid, spot_dict['E'], 0.0)
        cell_e = jnp.zeros(N_events * self.N_cells_tot)
        cell_e = cell_e.at[spot_global_cell_idx].add(spot_e)

        ### Add noise
        ### (TODO)

        ### Zero out cells below detection threshold
        if self.cell_e_threshold > 0:
            cell_e = jnp.where(cell_e < self.cell_e_threshold, 0.0, cell_e)

        grid_dict = {}
        if return_grid:
            ### Arrange energy deposits into grid (unflatten)
            grid_dict['cell_e'] = cell_e.reshape(N_events, self.N_cells[0], self.N_cells[1], self.N_cells[2])

        ### Fast return
        if not return_hits and not return_truth:
            return grid_dict

        ### Hits = active cells
        hit_valid = cell_e > 0
        hit_global_cell_idx = jnp.arange(len(cell_e))
        hit_event_idx = hit_global_cell_idx // self.N_cells_tot

        ### Compute local hit index within each event
        hit_valid_2d = hit_valid.reshape(N_events, self.N_cells_tot)
        hit_local_idx_2d = jnp.cumsum(hit_valid_2d, axis=1) - 1
        hit_local_idx = jnp.where(hit_valid, hit_local_idx_2d.flatten(), -1)

        hit_dict = {}
        if return_hits:

            ### Hit local cell index
            hit_local_cell_idx = hit_global_cell_idx % self.N_cells_tot

            ### Compute cell coordinates from local cell index
            hit_coords = self._local_cell_idx_to_coords(hit_local_cell_idx)

            hit_dict = {
                'hit_event_idx': jnp.where(hit_valid, hit_event_idx, -1),
                'hit_idx':       hit_local_idx, # already -1 for invalid hits
                'hit_cell_idx':  jnp.where(hit_valid, hit_local_cell_idx, -1),
                'hit_e':         jnp.where(hit_valid, cell_e[hit_global_cell_idx], 0.0),
                'hit_valid':     hit_valid
            }
            hit_dict[f'hit_{self.coords[0]}'] = jnp.where(hit_valid, hit_coords[0], -1)
            hit_dict[f'hit_{self.coords[1]}'] = jnp.where(hit_valid, hit_coords[1], -1)
            hit_dict[f'hit_{self.coords[2]}'] = jnp.where(hit_valid, hit_coords[2], -1)


        truth_dict = {}
        if return_truth:

            ### Compute index which tells which truth (particle, cell) pair to associate with each spot
            spot_global_truth_idx = jnp.where(spot_valid,
                                              spot_global_cell_idx * N_particles + spot_local_part_idx, 
                                              -1)

            ### Aggregate spots into flat (N_events * N_cells * N_particles) array of truth energy
            truth_e = jnp.zeros(N_events * self.N_cells_tot * N_particles)
            truth_e = truth_e.at[spot_global_truth_idx].add(spot_e)

            ### Validity
            truth_valid = truth_e > 0

            ### Compute local (event, cell, particle) indices corresponding to each nonzero truth entry
            truth_global_idx      = jnp.arange(len(truth_e))
            truth_event_idx       = (truth_global_idx // N_particles) // self.N_cells_tot
            truth_local_cell_idx  = (truth_global_idx // N_particles)  % self.N_cells_tot
            truth_local_part_idx  =  truth_global_idx                  % N_particles

            ### Compute global hit index for each cell (-1 if inactive/thresholded)
            cell_global_hit_idx = jnp.full((N_events * self.N_cells_tot,), -1, dtype=jnp.int32)
            cell_global_hit_idx = cell_global_hit_idx.at[hit_global_cell_idx].set(jnp.arange(len(hit_global_cell_idx)))

            ### Lookup local hit index for each truth entry
            truth_global_cell_idx = truth_event_idx * self.N_cells_tot + truth_local_cell_idx
            truth_local_hit_idx = hit_local_idx[truth_global_cell_idx]

            if self.cell_e_threshold > 0:
                ### Drop entries if E>0 but hit failed threshold
                truth_valid = truth_valid & (truth_local_hit_idx >= 0)

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


class CaloBlock(CaloBase):

    ### Cells on Cartesian grid

    def __init__(self, config):
        super().__init__(config)

        # N+1 bin edges covering the full range
        self.cell_edges[0] = jnp.linspace(-self.size[0]/2, self.size[0]/2, self.N_cells[0] + 1)
        self.cell_edges[1] = jnp.linspace(-self.size[1]/2, self.size[1]/2, self.N_cells[1] + 1)

    def _spots_to_local_cell_idx(self, spot_dict, particle_xs, particle_ys):

        flat_xs = particle_xs.reshape(-1)
        flat_ys = particle_ys.reshape(-1)

        ### Convert to Cartesian, adding per-particle offsets
        spot_x = spot_dict['r'] * jnp.cos(spot_dict['phi']) + flat_xs[spot_dict['particle_idx']]
        spot_y = spot_dict['r'] * jnp.sin(spot_dict['phi']) + flat_ys[spot_dict['particle_idx']]
        spot_z = spot_dict['t']

        ### Find the local cell index on (x,y,z) axes where the spot falls (floor-division on uniform grid)
        spot_local_cell_idx_x = ((spot_x - self.cell_edges[0][0]) / self.cell_sizes[0]).astype(jnp.int32).clip(0, self.N_cells[0] - 1)
        spot_local_cell_idx_y = ((spot_y - self.cell_edges[1][0]) / self.cell_sizes[1]).astype(jnp.int32).clip(0, self.N_cells[1] - 1)
        spot_local_cell_idx_z = ((spot_z - self.cell_edges[2][0]) / self.cell_sizes[2]).astype(jnp.int32).clip(0, self.N_cells[2] - 1)

        ### Combine these to get a single local cell index running over all N_cells in each event
        spot_local_cell_idx = spot_local_cell_idx_x * (self.N_cells[1] * self.N_cells[2]) \
                            + spot_local_cell_idx_y * self.N_cells[2] \
                            + spot_local_cell_idx_z

        ### Flag spots outside geometry as invalid with -1 index
        spot_contained = (spot_x >= -self.size[0]/2) & (spot_x < self.size[0]/2) & \
                         (spot_y >= -self.size[1]/2) & (spot_y < self.size[1]/2) & \
                         (spot_z >= 0)               & (spot_z < self.size[2])

        spot_local_cell_idx = jnp.where(spot_contained, spot_local_cell_idx, -1)

        return spot_local_cell_idx
    
    def _local_cell_idx_to_coords(self, local_cell_idx: jax.Array) -> tuple:

        ### Convert local cell index -> local cell index on (x,y,z) axes
        local_cell_idx_x =  local_cell_idx // (self.N_cells[1] * self.N_cells[2])
        local_cell_idx_y = (local_cell_idx  % (self.N_cells[1] * self.N_cells[2])) // self.N_cells[2]
        local_cell_idx_z =  local_cell_idx  %  self.N_cells[2]

        ### Compute corresponding cell centers
        x = self.cell_edges[0][0] + (local_cell_idx_x + 0.5) * self.cell_sizes[0]
        y = self.cell_edges[1][0] + (local_cell_idx_y + 0.5) * self.cell_sizes[1]
        z = self.cell_edges[2][0] + (local_cell_idx_z + 0.5) * self.cell_sizes[2]

        return x, y, z