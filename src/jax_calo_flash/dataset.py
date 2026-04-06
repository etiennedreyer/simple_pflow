import torch
from torch.utils.data import IterableDataset
from .calorimeter import CaloBlock
from .generator import EventGenerator
from .utils import transform, get_max_N_safe
from jax import dlpack as jdl
from torch.utils.dlpack import from_dlpack, to_dlpack
import yaml


class SimplePflowDataset(IterableDataset):
    
    def __init__(self, cfg, batch_size=None, device=None):
        super().__init__()

        if isinstance(cfg, str):
            with open(cfg) as f:
                cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.gen_cfg = cfg['generator']
        self.calo_cfg = cfg['calorimeter']
        self.xform_cfg = cfg['transforms']
        self.max_particles = self.gen_cfg['N_range'][1]
        if batch_size is None:
            self.batch_size = cfg.get('batch_size', 1)
        else:
            self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cpu')

    @staticmethod
    def get_incidence_matrix(hit_idx, particle_idx, weight, event_idx=None, 
                             N_particles=None, N_hits=None, cat_indicator=True):

        ### Support single event
        squeezed = False
        if event_idx is None:
            squeezed = True
            event_idx = torch.zeros_like(hit_idx)

        N_events = event_idx.max().item() + 1

        ### Determine shape of matrix
        N_particles = get_max_N_safe(particle_idx, N_particles)
        N_hits      = get_max_N_safe(hit_idx, N_hits)

        ### Energy deposition matrix
        energy_im = torch.zeros((N_events, N_particles, N_hits),
                                 dtype=torch.float32, device=hit_idx.device)
        energy_im[event_idx, particle_idx, hit_idx] = weight

        ### Normalize rows to sum to 1 (or 0 if no energy)
        cell_energies = energy_im.sum(dim=1, keepdim=True)
        im = energy_im / (cell_energies + 1e-8)

        particle_dep_energies = energy_im.sum(dim=2)
        if cat_indicator:
            ### Indicator (existence) column
            ind = (particle_dep_energies > 0).unsqueeze(2)
            im = torch.cat([im, ind.float()], dim=2)

        ### Remove event dim
        if squeezed:
             im.squeeze_(0)
             particle_dep_energies.squeeze_(0)

        return im, particle_dep_energies


    def __iter__(self):

        generator = EventGenerator(self.gen_cfg)
        calorimeter = CaloBlock(self.calo_cfg)

        while True:
            p_E, p_x, p_y = generator.generate(self.batch_size)
            calo_dict = calorimeter.simulate(p_E, p_x, p_y,
                                          return_grid=False, 
                                          return_hits=True, 
                                          return_truth=True)

            ### Convert to torch tensors
            p_E = from_dlpack(jdl.to_dlpack(p_E))
            p_x = from_dlpack(jdl.to_dlpack(p_x))
            p_y = from_dlpack(jdl.to_dlpack(p_y))
            calo_dict = {k: from_dlpack(jdl.to_dlpack(v)) for k, v in calo_dict.items()}

            ### Incidence matrix
            N_hits = calo_dict['hit_idx'].max().item() + 1
            inc, part_dep_E = self.get_incidence_matrix(
                hit_idx=calo_dict['truth_hit_idx'],
                particle_idx=calo_dict['truth_particle_idx'],
                weight=calo_dict['truth_e'],
                event_idx=calo_dict['truth_event_idx'],
                N_particles=self.max_particles,
                N_hits=N_hits
            )


            ### Features
            input_feats = {k: v for k, v in calo_dict.items() \
                          if k in ['hit_x', 'hit_y', 'hit_z', 'hit_e']}

            target_feats = {
                'part_e': p_E,
                'part_x': p_x,
                'part_y': p_y
            }

            ### NaN-pad particles to max_particles (no-op usually)
            how_much = self.max_particles - p_E.shape[1]
            if how_much > 0:
                nans = torch.full((p_E.shape[0], how_much), float('nan'), device=p_E.device)
                target_feats = {k: torch.cat([v, nans], dim=1) for k, v in target_feats.items()}

            ### NaN-pad particles with no deposited E
            target_mask = (part_dep_E == 0)
            for k, v in target_feats.items():
                v[target_mask] = float('nan')

            ### Transform features
            input_mask = ~calo_dict['hit_valid']
            input_feats  = {k: transform(v, k, self.xform_cfg, mask=input_mask) \
                            for k, v in input_feats.items()}
            target_feats = {k: transform(v, k, self.xform_cfg, mask=target_mask) \
                            for k, v in target_feats.items()}

            if self.batch_size > 1:
                ### Unflatten input into padded tensors
                hit_evt_idx = calo_dict['hit_event_idx']
                hit_loc_idx = calo_dict['hit_idx']
                for k, v in input_feats.items():
                    unflat = torch.full((self.batch_size, N_hits), float('nan'), 
                                        dtype=v.dtype, device=v.device)
                    unflat[hit_evt_idx, hit_loc_idx] = v
                    input_feats[k] = unflat

            yield {
                'input_feats': input_feats,
                'target_feats': target_feats,
                'incidence_matrix': inc
            }
            
