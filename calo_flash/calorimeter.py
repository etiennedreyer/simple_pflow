import torch
import numpy as np
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

        if device is None:
            self.get_device()

        self.initialize_cells()

    def get_device(self):
        if torch.cuda.is_available():
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
                 store_truth=True, N_spots_per_layer=None):

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

        ### Deposit energy into (N_events, N_cells_x, N_cells_y, N_cells_z)
        flat_e = torch.zeros(N_events * self.N_cells, device=self.device)
        flat_e.scatter_add_(0, global_cell_idx, e_m)
        flat_e = flat_e.reshape(N_events, self.N_cells_x, self.N_cells_y, self.N_cells_z)

        ### remove event dimension for single-event case
        if N_events == 1:
            flat_e = flat_e[0]

        if not store_truth:
            ### Save some time
            return flat_e

        ### Truth record: unique key for (event, cell, particle)
        combined_idx = global_cell_idx * N_particles + particle_m
        wgt_dense = torch.zeros(N_events * self.N_cells * N_particles, device=self.device)
        wgt_dense.scatter_add_(0, combined_idx, e_m)
        nonzero = wgt_dense.nonzero(as_tuple=True)[0]

        ### Invert combined index
        event_cell = nonzero // N_particles
        cell_event_src    = (event_cell // self.N_cells).tolist()
        cell_particle_src = (event_cell  % self.N_cells).tolist()
        cell_particle_dst = (nonzero     % N_particles).tolist()
        cell_particle_wgt = wgt_dense[nonzero].tolist()

        truth = {
            'cell_event_src': cell_event_src,
            'cell_particle_src': cell_particle_src,
            'cell_particle_dst': cell_particle_dst,
            'cell_particle_wgt': cell_particle_wgt
        }

        return flat_e, truth

class EventGenerator:

    def __init__(self, config):
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']

    def generate(self, N_events=1):

        N_particles = torch.randint(self.N_min, self.N_max + 1, (N_events,)).tolist()

        N_pad = max(N_particles)

        ### Uniform spread in x/y
        particle_xs = torch.rand((N_events, N_pad)) * (self.x_max - self.x_min) + self.x_min
        particle_ys = torch.rand((N_events, N_pad)) * (self.y_max - self.y_min) + self.y_min

        ### Power-law spectrum in energy
        alpha = 2.0
        r = torch.rand((N_events, N_pad))
        particle_Es = ((self.E_max**(1-alpha) - self.E_min**(1-alpha)) * r + self.E_min**(1-alpha))**(1/(1-alpha))

        if N_events == 1:
            return particle_Es[0], particle_xs[0], particle_ys[0]

        ### Mask out padded particles, vectorized
        for i, N in enumerate(N_particles):
            particle_Es[i, N:] = 0.0  # zero energy means "no particle"

        return particle_Es, particle_xs, particle_ys