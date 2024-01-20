#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:19:59
# @Author: zzm

import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from tensorboardX import SummaryWriter

from code.models import MGA
from code.dataset import get_loader
from code.config import get_parser

def train(args, logger, train_dataloader, clipmodel):
    optimizer = AdamW(clipmodel.parameters(), lr=1e-6)
    step = 0
    for epoch in range(args.n_epoch):
        for batch in tqdm(train_dataloader):
            step += 1
            sketch_id, image, sketch, txt, cate, tokens, masks = batch
            image, sketch, txt, cate, tokens, masks = image.cuda(), sketch.cuda(), txt.cuda(), cate.cuda(), tokens.cuda(), masks.cuda()
            
            Le_loss, Lc_loss, Ld_loss = clipmodel(image, txt, sketch, cate, tokens, masks)
            total_loss = (10 * Lc_loss + Ld_loss + 100 * Le_loss) / 111
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            logger.add_scalar("epoch", epoch, step)
            if step % 100 == 0:
                print('[%d / %d] loss: %.10f' %(epoch, step, total_loss))
                logger.add_scalar("loss/le_loss", Le_loss.detach().cpu().numpy(), step)
                logger.add_scalar("loss/lc_loss", Lc_loss.detach().cpu().numpy(), step)
                logger.add_scalar("loss/ld_loss", Ld_loss.detach().cpu().numpy(), step)
                logger.add_scalar("loss/total_loss", total_loss.detach().cpu().numpy(), step)
                torch.save({
                    'epoch':epoch,
                    'opt':args,
                    'state_dict':clipmodel.state_dict()},
                    logger.file_writer.get_logdir()+'/latest_checkpoint.pth'
                )
            
    torch.save({
            'epoch':epoch,
            'opt':args,
            'state_dict':clipmodel.state_dict()},
            logger.file_writer.get_logdir()+'/latest_checkpoint.pth'
        )

if __name__ == '__main__':
    parser = get_parser(split='train')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_dataloader = get_loader(args, 'train')

    logger = SummaryWriter(comment=args.logger_comment)
    print('Log files saved to', logger.file_writer.get_logdir())
    for k in list(args.__dict__.keys()):
        logger.add_text(k, str(args.__dict__[k]))

    # load clip model
    model_config_file = './model_pt/ViT-B-16.json'
    model_file = './model_pt/tsbir_model_final.pt'
    # model_file = './runs/Dec08_12-29-07_dp3090tsbir_sketchycocolf/latest_checkpoint.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = MGA(args, **model_info)
    checkpoints = torch.load(model_file, map_location='cpu')
    sd = checkpoints["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.train()
    clipmodel = model.to(device)

    train(args, logger, train_dataloader, clipmodel)
    logger.close()

'''
python train.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD-open --logger_comment tsbir_SFSD_sketch_text
python train.py --dataset FScocoDataset --dataset_root_path ~/datasets/fscoco --output_dim 80 --logger_comment tsbir_fscoco
python train.py --dataset SketchycocoDataset --dataset_root_path ~/datasets/SketchyCOCO --output_dim 80 --logger_comment tsbir_sketchycoco
python train.py --dataset SketchycocoLFDataset --dataset_root_path ~/datasets/SketchyCOCO-lf --output_dim 80 --logger_comment tsbir_sketchycocolf
'''