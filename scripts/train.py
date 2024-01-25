from code.datasets import get_loader
from code.models import get_model
from code.config import get_parser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pprint import pprint 

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pprint(args)

    dataloaders = dict()
    for split in ['train', 'test']:
        dataloaders[split] = get_loader(args, split)
    
    model = get_model(args)
    
    logger = TensorBoardLogger(save_dir = args.logger_path,
                                name = args.logger_comment,
                                default_hp_metric = False)  
    
    # 参考https://blog.csdn.net/qq_42363032/article/details/134010353
    best_callback = ModelCheckpoint(
        monitor="val/top1",
        mode='max',
        filename='best-{}'.format(args.logger_comment),  # 文件名格式
        save_top_k=1,
        auto_insert_metric_name=True 
    )

    last_callback = ModelCheckpoint(
        filename='last-{}'.format(args.logger_comment),
        save_top_k=1,
        auto_insert_metric_name=True 
    )

    trainner = pl.Trainer(gpus = args.gpu_nums,
                          accelerator = "gpu",
                          max_epochs = args.n_epoch,
                          sync_batchnorm = False,
                          num_sanity_val_steps = 0,
                          logger = logger,
                          callbacks = [best_callback, last_callback]
                          )
    trainner.fit(model, dataloaders['train'], dataloaders['test'])

'''
python scripts/train.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD-open --logger_comment SFSD
python scripts/train.py --dataset FScocoDataset --dataset_root_path ~/datasets/fscoco --logger_comment FSCOCO --output_dim 80
python scripts/train.py --dataset SketchycocoDataset --dataset_root_path ~/datasets/SketchyCOCO --logger_comment SketchyCOCO --output_dim 80
python scripts/train.py --dataset SketchycocoLFDataset --dataset_root_path ~/datasets/SketchyCOCO-lf --logger_comment SketchyCOCOlf --output_dim 80 --resume ./runs/SketchyCOCO/version_0/checkpoints/best-SketchyCOCO.ckpt --pre_model False
'''