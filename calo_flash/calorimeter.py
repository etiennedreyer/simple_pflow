import numpy as np
import calo_flash

class CaloBlock:

    ### Homogeneous calorimeter block with cells on Cartesian grid

    def __init__(self, config):
        self.Z = config['Z']
        self.width = config['width']
        self.height = config.get('height', self.width)
        self.depth = config['depth']
        self.N_cells_x = config['N_cells_x']
        self.N_cells_y = config.get('N_cells_y', self.N_cells_x)
        self.N_cells_z = config['N_cells_z']
        self.N_cells = self.N_cells_x * self.N_cells_y * self.N_cells_z

        self.initialize_cells()

    def initialize_cells(self):
        self.cell_size_x = self.width / self.N_cells_x
        self.cell_size_y = self.height / self.N_cells_y
        self.cell_size_z = self.depth / self.N_cells_z

        self.cell_x = np.linspace(-self.width/2, self.width/2, self.N_cells_x)
        self.cell_y = np.linspace(-self.height/2, self.height/2, self.N_cells_y)
        self.cell_z = np.linspace(0, self.depth, self.N_cells_z)

        # Bin edges used for cell lookup (N+1 edges covering the full range)
        self.cell_x_edges = np.linspace(-self.width/2, self.width/2, self.N_cells_x + 1)
        self.cell_y_edges = np.linspace(-self.height/2, self.height/2, self.N_cells_y + 1)
        self.cell_z_edges = np.linspace(0, self.depth, self.N_cells_z + 1)
        self.cell_e = np.zeros((self.N_cells_x, self.N_cells_y, self.N_cells_z))
        self.cell_local_indices = np.indices(self.cell_e.shape).reshape(3, -1).T
        self.cell_global_indices = np.arange(self.N_cells).reshape(self.cell_e.shape)

        ### For storing flattened truth record
        self.cell_particle_src = []
        self.cell_particle_dst = []
        self.cell_particle_wgt = []
    
    def clear(self):
        self.cell_e.fill(0)
        self.cell_particle_src.clear()
        self.cell_particle_dst.clear()
        self.cell_particle_wgt.clear()

    def shower(self, particle_E, 
               particle_x=0, particle_y=0, particle_idx=0):

        ### Calo Flash simulation
        spots = calo_flash.shoot(particle_E, self.Z, self.cell_z)

        ### Convert to Cartesian coordinates
        x = spots['r'] * np.cos(spots['phi']) + particle_x
        y = spots['r'] * np.sin(spots['phi']) + particle_y
        z = spots['t']
        e = spots['E']

        ### Lookup cells using bin edges so no cell accumulates out-of-range deposits
        idx_x = np.searchsorted(self.cell_x_edges, x, side='right') - 1
        idx_y = np.searchsorted(self.cell_y_edges, y, side='right') - 1
        idx_z = np.searchsorted(self.cell_z_edges, z, side='right') - 1

        ### Mask for valid indices
        mask = (idx_x >= 0) & (idx_x < self.N_cells_x) & \
               (idx_y >= 0) & (idx_y < self.N_cells_y) & \
               (idx_z >= 0) & (idx_z < self.N_cells_z)
        idx_x = idx_x[mask]
        idx_y = idx_y[mask]
        idx_z = idx_z[mask]
        e = e[mask]

        ### Deposit energy in cells
        np.add.at(self.cell_e, (idx_x, idx_y, idx_z), e)

        ### Link these cells to the particle in the truth record
        global_indices = self.cell_global_indices[idx_x, idx_y, idx_z]
        self.cell_particle_src.extend(global_indices.tolist())
        self.cell_particle_dst.extend([particle_idx] * len(global_indices))
        self.cell_particle_wgt.extend(e.tolist())

    def simulate(self, particle_Es, particle_xs, particle_ys):
        self.clear()

        for i, (E, x, y) in enumerate(zip(particle_Es, particle_xs, particle_ys)):
            self.shower(E, particle_x=x, particle_y=y, particle_idx=i)


class EventGenerator:

    def __init__(self, config):
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']

    def generate(self):

        N = np.random.randint(self.N_min, self.N_max + 1)

        ### Uniform spread in x/y
        particle_xs = np.random.uniform(self.x_min, self.x_max, N)
        particle_ys = np.random.uniform(self.y_min, self.y_max, N)
        
        ### Power-law spectrum in energy
        alpha = 2.0
        r = np.random.uniform(0, 1, N)
        particle_Es = ((self.E_max**(1-alpha) - self.E_min**(1-alpha)) * r + self.E_min**(1-alpha))**(1/(1-alpha))

        return particle_Es, particle_xs, particle_ys