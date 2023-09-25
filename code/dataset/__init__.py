from .SFSDDataset import SFSDDataset
from .SketchycocoDataset import SketchycocoDataset
from .FScocoDataset import FScocoDataset

import torch.utils.data as td

def get_dataset(config, split='train'):
    return globals()[config.dataset](config, split)

def get_loader(config, split='train'):
    dataset = get_dataset(config, split)
    
    loader = td.DataLoader(dataset,
                  batch_size = config.batch_size,
                  shuffle = True if split == 'train' else False,
                  num_workers = config.loader_num_workers,
                  drop_last = False)
    return loader