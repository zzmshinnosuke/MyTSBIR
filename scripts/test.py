from code.datasets import get_loader
from code.models import get_model
from code.config import get_parser

import pytorch_lightning as pl
from pprint import pprint 

if __name__ == '__main__':
    parser = get_parser(split='test')
    args = parser.parse_args()
    pprint(args)

    dataloader = get_loader(args, 'test')
    
    model = get_model(args)

    trainner = pl.Trainer(gpus = 1,
                          max_epochs = 1,
                          sync_batchnorm = False,
                          logger = False)

    trainner.test(model, dataloader)

#python scripts/test.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD-open


