import torch
from torch.utils.data import IterableDataset, get_worker_info
from calorimeter import CaloBlock
from generator import EventGenerator
from utils import transform, unflatten
import yaml


class SimplePflowDataset(IterableDataset):
    
    def __init__(self, cfg, batch_size=1, device=None):
        super().__init__()

        if isinstance(cfg, str):
            with open(cfg) as f:
                cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.gen_cfg = cfg['generator']
        self.calo_cfg = cfg['calorimeter']
        self.xform_cfg = cfg['transforms']
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cpu')

    def incidence_matrix(self, output, N_particles):
        evt_idx = output['truth_event_idx']
        c_p_src = output['truth_cell_particle_src']
        c_p_dst = output['truth_cell_particle_dst']
        c_p_wgt = output['truth_cell_particle_e']

        N_cells = c_p_src.max().item() + 1
        energy_im = torch.zeros((self.batch_size, N_particles, N_cells),
                                 dtype=torch.float32, device=c_p_src.device)

        energy_im[evt_idx, c_p_dst, c_p_src] = c_p_wgt
        
        ### Normalize rows to sum to 1 (or 0 if no energy)
        cell_energies = energy_im.sum(dim=1, keepdim=True)
        im = energy_im / (cell_energies + 1e-8)

        return im


    def __iter__(self):

        generator = EventGenerator(self.gen_cfg)
        calorimeter = CaloBlock(self.calo_cfg, device=self.device)

        while True:
            p_E, p_x, p_y = generator.generate(self.batch_size)
            output = calorimeter.simulate(p_E, p_x, p_y,
                                          return_grid=False, 
                                          return_point_cloud=True, 
                                          return_truth=True)

            ### Features            
            input_feats = {k: v for k, v in output.items() \
                          if k in ['cell_x', 'cell_y', 'cell_z', 'cell_e']}

            target_mask = p_E == 0
            target_feats = {
                'particle_E': p_E,
                'particle_x': p_x,
                'particle_y': p_y
            }

            ### NaN-pad target features
            for k, v in target_feats.items():
                v[target_mask] = float('nan')

            ### Transform features
            input_feats = {k: transform(v, k, self.xform_cfg) for k, v in input_feats.items()}
            target_feats = {k: transform(v, k, self.xform_cfg) for k, v in target_feats.items()}

            if self.batch_size > 1:
                ### Unflatten input into padded tensors
                idx_0 = output['event_idx']
                idx_1 = None
                for k, v in input_feats.items():
                    input_feats[k], idx_1 = unflatten(v, idx_0, idx_1)

            ### Incidence matrix
            N_particles = p_E.shape[0] if p_E.dim() == 1 else p_E.shape[1]
            target_feats['incidence_matrix'] = self.incidence_matrix(output, N_particles)

            yield input_feats, target_feats
            
