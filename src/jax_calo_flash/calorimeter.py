from functools import partial
from math import prod
import yaml

import jax
import jax.numpy as jnp
from .calo_flash import shoot
from functools import partial
from typing import Union, Tuple

class CaloBase:

    ### Homogeneous calorimeter simulation with no geometry specified for now

    def __init__(self, config: Union [str, dict]):
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
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

    def _spots_to_local_cell_idx(self, spot_dict: dict, 
                                 particle_xs: Union [jax.Array, None] = None,
                                 particle_ys: Union [jax.Array, None] = None
                                 ) -> jax.Array:

        ### Find the local cell index on (var0,var1,z) axes where the spot falls (floor-division on uniform grid)
        vars = self.coords[:2]
        spot_local_cell_idx_var0 = ((spot_dict[vars[0]] - self.cell_edges[0][0]) / self.cell_sizes[0]).astype(jnp.int32).clip(0, self.N_cells[0] - 1)
        spot_local_cell_idx_var1 = ((spot_dict[vars[1]] - self.cell_edges[1][0]) / self.cell_sizes[1]).astype(jnp.int32).clip(0, self.N_cells[1] - 1)
        spot_local_cell_idx_z    = ((spot_dict['z']     - self.cell_edges[2][0]) / self.cell_sizes[2]).astype(jnp.int32).clip(0, self.N_cells[2] - 1)

        ### Combine these to get a single local cell index running over all N_cells in each event
        spot_local_cell_idx = spot_local_cell_idx_var0 * (self.N_cells[1] * self.N_cells[2]) \
                            + spot_local_cell_idx_var1 *  self.N_cells[2] \
                            + spot_local_cell_idx_z

        spot_local_cell_idx = jnp.where(spot_dict['contained'], spot_local_cell_idx, -1)

        return spot_local_cell_idx


    def _local_cell_idx_to_coords(self, local_cell_idx: jax.Array) -> tuple:
        pass

    @partial(jax.jit, static_argnames=['self', 'return_grid', 'return_hits', 'return_truth', 'N_spots_per_layer'])
    def simulate(self, particle_Es: jax.Array, 
                 particle_xs: jax.Array = None, particle_ys: jax.Array = None,
                 return_grid=True, return_hits=True, return_truth=True, 
                 N_spots_per_layer=None):

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

        ### Particle info dict
        particle_dict = {'particle_e': particle_Es}
        if particle_xs is not None:
            particle_dict['particle_x'] = particle_xs
        if particle_ys is not None:
            particle_dict['particle_y'] = particle_ys

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
            return {**particle_dict, **grid_dict}

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

        return {**particle_dict,
                **grid_dict, 
                **hit_dict, 
                **truth_dict}


class CaloBlock(CaloBase):

    ### Cells on Cartesian grid

    def __init__(self, config):
        super().__init__(config)
        assert self.coords == ['x', 'y', 'z'], "CaloBlock expects geometry {x:..., y:..., z:...}"

        self.cell_edges[0] = jnp.linspace(-self.size[0]/2, self.size[0]/2, self.N_cells[0] + 1)
        self.cell_edges[1] = jnp.linspace(-self.size[1]/2, self.size[1]/2, self.N_cells[1] + 1)

    def _spots_to_local_cell_idx(self, spot_dict, particle_xs=None, particle_ys=None):

        if particle_xs is None:
            flat_xs = jnp.zeros_like(spot_dict['r'])
        else:
            flat_xs = particle_xs.reshape(-1)

        if particle_ys is None:
            flat_ys = jnp.zeros_like(spot_dict['r'])
        else:
            flat_ys = particle_ys.reshape(-1)

        ### Convert to Cartesian, adding per-particle offsets
        spot_dict['x'] = spot_dict['r'] * jnp.cos(spot_dict['phi']) + flat_xs[spot_dict['particle_idx']]
        spot_dict['y'] = spot_dict['r'] * jnp.sin(spot_dict['phi']) + flat_ys[spot_dict['particle_idx']]
        spot_dict['z'] = spot_dict['t']

        ### Flag spots outside geometry as invalid with -1 index
        spot_dict['contained'] = (spot_dict['x'] >= -self.size[0]/2) & (spot_dict['x'] < self.size[0]/2) & \
                                 (spot_dict['y'] >= -self.size[1]/2) & (spot_dict['y'] < self.size[1]/2) & \
                                 (spot_dict['z'] >= 0)               & (spot_dict['z'] < self.size[2])

        spot_local_cell_idx = super()._spots_to_local_cell_idx(spot_dict, particle_xs=None, particle_ys=None)

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
    

class CaloCylinder(CaloBase):

    ### Cells on cylindrical grid

    def __init__(self, config):
        super().__init__(config)
        assert self.coords == ['r', 'phi', 'z'], "CaloCylinder expects geometry {r:..., phi:..., z:...}"
        self.size[1] = 2 * jnp.pi

        ### Only uniform binning supported for now
        self.cell_edges[0] = jnp.linspace(0, self.size[0], self.N_cells[0] + 1)
        self.cell_edges[1] = jnp.linspace(0, self.size[1], self.N_cells[1] + 1)

    def _spots_to_local_cell_idx(self, spot_dict, particle_xs=None, particle_ys=None):

        assert particle_xs is None and particle_ys is None, "CaloCylinder does not support per-particle offsets"

        spot_dict['z'] = spot_dict['t']

        ### Flag spots outside geometry as invalid with -1 index
        spot_dict['contained'] = (spot_dict['r']   >= 0) & (spot_dict['r']   < self.size[0]) & \
                                 (spot_dict['phi'] >= 0) & (spot_dict['phi'] < self.size[1]) & \
                                 (spot_dict['z']   >= 0) & (spot_dict['z']   < self.size[2])
        
        spot_local_cell_idx = super()._spots_to_local_cell_idx(spot_dict, particle_xs=None, particle_ys=None)

        return spot_local_cell_idx

    def _local_cell_idx_to_coords(self, local_cell_idx: jax.Array) -> tuple:

        ### Convert local cell index -> local cell index on (r,phi,z) axes
        local_cell_idx_r   =  local_cell_idx // (self.N_cells[1] * self.N_cells[2])
        local_cell_idx_phi = (local_cell_idx  % (self.N_cells[1] * self.N_cells[2])) // self.N_cells[2]
        local_cell_idx_z   =  local_cell_idx  %  self.N_cells[2]

        ### Compute corresponding cell centers
        r   = self.cell_edges[0][0] + (local_cell_idx_r   + 0.5) * self.cell_sizes[0]
        phi = self.cell_edges[1][0] + (local_cell_idx_phi + 0.5) * self.cell_sizes[1]
        z   = self.cell_edges[2][0] + (local_cell_idx_z   + 0.5) * self.cell_sizes[2]

        return r, phi, z
