import torch
from torch.utils.data import IterableDataset, get_worker_info
from calorimeter import CaloBlock
from generator import EventGenerator
import yaml

class SimplePflowDataset(IterableDataset):
    
    def __init__(self, cfg, batch_size, device=None):
        super().__init__()

        if isinstance(cfg, str):
            with open(cfg) as f:
                cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.gen_cfg = cfg['generator']
        self.calo_cfg = cfg['calorimeter']
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cpu')

    def __iter__(self):

        calorimeter = CaloBlock(self.calo_cfg, device=self.device)
        generator = EventGenerator(self.gen_cfg)

        while True:
            events = generator.generate(self.batch_size)
            output = calorimeter.simulate(*events, 
                                        return_grid=False, 
                                        return_point_cloud=True, 
                                        return_truth=True)
            yield output
