import torch
from calo_flash import shoot

class CaloBlock:

    ### Homogeneous calorimeter block with cells on Cartesian grid

    def __init__(self, config, device=None):
        self.Z = config['Z']
        self.width = config['width']
        self.height = config.get('height', self.width)
        self.depth = config['depth']
        self.N_cells_x = config['N_cells_x']
        self.N_cells_y = config.get('N_cells_y', self.N_cells_x)
        self.N_cells_z = config['N_cells_z']
        self.N_cells = self.N_cells_x * self.N_cells_y * self.N_cells_z
        self.N_spots_per_layer = config.get('N_spots_per_layer', 1000)

        self.get_device(device)
        self.initialize_cells()

    def get_device(self, device=None):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def initialize_cells(self):
        self.cell_size_x = self.width / self.N_cells_x
        self.cell_size_y = self.height / self.N_cells_y
        self.cell_size_z = self.depth / self.N_cells_z

        # N+1 bin edges covering the full range
        self.cell_x_edges = torch.linspace(-self.width/2, self.width/2, self.N_cells_x + 1, device=self.device)
        self.cell_y_edges = torch.linspace(-self.height/2, self.height/2, self.N_cells_y + 1, device=self.device)
        self.cell_z_edges = torch.linspace(0, self.depth, self.N_cells_z + 1, device=self.device)

    def simulate(self, particle_Es: torch.Tensor, particle_xs: torch.Tensor, particle_ys: torch.Tensor,
                 return_grid=True, return_point_cloud=True, return_truth=True, N_spots_per_layer=None):

        ### Auto-unsqueeze (N_particles,) -> (1, N_particles)
        if particle_Es.dim() == 1:
            particle_Es = particle_Es.unsqueeze(0)
            particle_xs = particle_xs.unsqueeze(0)
            particle_ys = particle_ys.unsqueeze(0)

        ### Shapes
        N_events, N_particles = particle_Es.shape
        self.max_particles = N_particles

        ### Move to device
        particle_Es = particle_Es.to(device=self.device, dtype=torch.float32)
        particle_xs = particle_xs.to(device=self.device, dtype=torch.float32)
        particle_ys = particle_ys.to(device=self.device, dtype=torch.float32)

        ### Flatten: (N_events, N_particles) -> (N_events * N_particles,)
        flat_Es = particle_Es.reshape(-1)
        flat_xs = particle_xs.reshape(-1)
        flat_ys = particle_ys.reshape(-1)

        ### Mask out padded (E == 0) particles
        valid_mask    = flat_Es > 0
        valid_Es      = flat_Es[valid_mask]
        orig_flat_idx = valid_mask.nonzero(as_tuple=True)[0]  # indices into flat_Es

        ### Calo Flash simulation (only valid particles)
        if N_spots_per_layer is None:
            N_spots_per_layer = self.N_spots_per_layer
        spots = shoot(valid_Es, self.Z, self.cell_z_edges, N_spots_per_layer=N_spots_per_layer)

        ### Map spot particle index -> flat global index -> (event, local particle)
        local_pidx   = spots['particle_idx'].long()
        flat_pidx    = orig_flat_idx[local_pidx]          # index into flat_Es
        event_idx    = flat_pidx // N_particles
        particle_idx = flat_pidx  % N_particles

        ### Convert to Cartesian
        x = spots['r'] * torch.cos(spots['phi']) + flat_xs[flat_pidx]
        y = spots['r'] * torch.sin(spots['phi']) + flat_ys[flat_pidx]
        z = spots['t']
        e = spots['E']

        ### Discard out-of-bounds spots
        mask = (x >= -self.width/2)  & (x < self.width/2) & \
               (y >= -self.height/2) & (y < self.height/2) & \
               (z >= 0)              & (z < self.depth)
        x_m, y_m, z_m, e_m = x[mask], y[mask], z[mask], e[mask]
        event_m, particle_m = event_idx[mask], particle_idx[mask]

        ### Cell indices (floor-division on uniform grid)
        ix = ((x_m - self.cell_x_edges[0]) / self.cell_size_x).long().clamp(0, self.N_cells_x - 1)
        iy = ((y_m - self.cell_y_edges[0]) / self.cell_size_y).long().clamp(0, self.N_cells_y - 1)
        iz = ((z_m - self.cell_z_edges[0]) / self.cell_size_z).long().clamp(0, self.N_cells_z - 1)
        cell_idx = ix * (self.N_cells_y * self.N_cells_z) + iy * self.N_cells_z + iz

        ### Global index in (N_events x N_cells) flattened tensor
        global_cell_idx = event_m * self.N_cells + cell_idx

        ### Aggregate energy into (N_events, N_cells)
        flat_cell_e = torch.zeros(N_events * self.N_cells, device=self.device)
        flat_cell_e.scatter_add_(0, global_cell_idx, e_m)

        grid_dict = {}
        if return_grid:
            ### Arrange energy deposits into grid (events, cells_x, cells_y, cells_z)
            grid_e = flat_cell_e.reshape(N_events, self.N_cells_x, self.N_cells_y, self.N_cells_z)

            ### remove event dimension for single-event case
            if N_events == 1: grid_e = grid_e[0]

            grid_dict['grid_e'] = grid_e

        point_dict = {}
        if return_point_cloud:

            active_cell_idx = flat_cell_e.nonzero(as_tuple=True)[0]
            hit_event = active_cell_idx // self.N_cells
            hit_cell  = active_cell_idx  % self.N_cells

            ### Convert global -> local cell index
            hit_ix =  hit_cell // (self.N_cells_y * self.N_cells_z)
            hit_iy = (hit_cell  % (self.N_cells_y * self.N_cells_z)) // self.N_cells_z
            hit_iz =  hit_cell  %  self.N_cells_z

            ### Compute corresponding cell centers
            hit_x = self.cell_x_edges[0] + (hit_ix + 0.5) * self.cell_size_x
            hit_y = self.cell_y_edges[0] + (hit_iy + 0.5) * self.cell_size_y
            hit_z = self.cell_z_edges[0] + (hit_iz + 0.5) * self.cell_size_z
            hit_e = flat_cell_e[active_cell_idx]

            point_dict = {
                'event_idx': hit_event,
                'cell_x': hit_x,
                'cell_y': hit_y,
                'cell_z': hit_z,
                'cell_e': hit_e
            }

        truth_dict = {}
        if return_truth:

            ### Truth record: unique key for (event, cell, particle)
            combined_idx = global_cell_idx * N_particles + particle_m
            wgt_dense = torch.zeros(N_events * self.N_cells * N_particles, device=self.device)
            wgt_dense.scatter_add_(0, combined_idx, e_m)
            nonzero = wgt_dense.nonzero(as_tuple=True)[0]

            ### Invert combined index
            event_idx         = (nonzero // N_particles) // self.N_cells
            cell_particle_src = (nonzero // N_particles)  % self.N_cells
            cell_particle_dst =  nonzero                  % N_particles
            cell_particle_wgt = wgt_dense[nonzero]

            truth_dict = {
                'truth_event_idx':         event_idx,
                'truth_cell_particle_src': cell_particle_src,
                'truth_cell_particle_dst': cell_particle_dst,
                'truth_cell_particle_e':   cell_particle_wgt
            }

        return {**grid_dict, 
                **point_dict, 
                **truth_dict}