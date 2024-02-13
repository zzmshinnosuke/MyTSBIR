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

# python scripts/test.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD-open --resume ./runs/SFSD/version_7/checkpoints/best-SFSD.ckpt --result_path ./results
# python scripts/test.py --dataset FScocoDataset --dataset_root_path ~/datasets/fscoco --output_dim 80 --resume ./runs/FSCOCO/version_1/checkpoints/best-FSCOCO.ckpt --result_path ./results
# python scripts/test.py --dataset SketchycocoDataset --dataset_root_path ~/datasets/SketchyCOCO --output_dim 80 --resume ./runs/SketchyCOCO/version_0/checkpoints/best-SketchyCOCO.ckpt --result_path ./results
# python scripts/test.py --dataset SketchycocoLFDataset --dataset_root_path ~/datasets/SketchyCOCO-lf --output_dim 80 --resume ./runs/SketchyCOCOlf/version_1/checkpoints/best-SketchyCOCOlf.ckpt --result_path ./results



