#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:19:59
# @Author: zzm

import os
import json
from collections import namedtuple
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.optim import AdamW
from tensorboardX import SummaryWriter

from transformers.models.gpt2.modeling_gpt2 import GPT2Config

from code.clip import tokenize, convert_weights, CLIP, ClassModel
from code.AsymmetricLoss import AsymmetricLossOptimized
from code.gpt import GPT2LMHeadModel
from code.dataset import get_loader
from code.config import get_parser

def train(args, logger, train_dataloader, clipmodel, gptmodel, classmodel):
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt_sketch = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(clipmodel.parameters(), lr=1e-6)
    ASL_Loss = AsymmetricLossOptimized()
    step = 0
    for epoch in range(args.n_epoch):
        for batch in tqdm(train_dataloader):
            step += 1
            image, sketch, txt, cate, tokens, masks = batch
            image, sketch, txt, cate, tokens, masks = image.cuda(), sketch.cuda(), txt.cuda(), cate.cuda(), tokens.cuda(), masks.cuda()
            
            image_feature, fused_feature = clipmodel(image, txt, sketch)
            sketch_feature = clipmodel.encode_sketch(sketch)
            text_feature = clipmodel.encode_text(txt)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)

            #Le
            logit_scale = clipmodel.logit_scale.exp()
            logits_per_image = logit_scale * image_feature @ fused_feature.t()
            logits_per_fuse = logits_per_image.t()
            if device == "cpu":
                ground_truth = torch.arange(batch[0].shape[0]).long().to(device)
            else:
                ground_truth = torch.arange(batch[0].shape[0], dtype=torch.long, device=device)
            Le_loss = (loss_img(logits_per_image, ground_truth) + loss_txt_sketch(logits_per_fuse, ground_truth)) / 2

            #Lc
            logit_txt = classmodel(text_feature.float())
            logit_img = classmodel(image_feature.float())
            logit_sketch = classmodel(sketch_feature.float())
            Lc_loss_txt = ASL_Loss(logit_txt, cate)
            Lc_loss_img = ASL_Loss(logit_img, cate)
            Lc_loss_sketch = ASL_Loss(logit_sketch, cate)
            Lc_loss = (Lc_loss_txt + Lc_loss_img + Lc_loss_sketch) / (3 * args.batch_size)       
            
            #Ld
            # with torch.no_grad():
            Ld_loss, outputs, _ = gptmodel(tokens, fused_feature, labels=tokens, attention_mask=masks)

            total_loss = (10 * Lc_loss + Ld_loss + 100 * Le_loss) / 111
            
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                optimizer.step()
                convert_weights(clipmodel)
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
                    'model_state_dict':clipmodel.state_dict()},
                    logger.file_writer.get_logdir()+'/latest_checkpoint.pth'
                )
            
    torch.save({
            'epoch':epoch,
            'opt':args,
            'model_state_dict':clipmodel.state_dict()},
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
    model_config_file = './code/training/model_configs/ViT-B-16.json'
    model_file = './model_pt/tsbir_model_final.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info)
    checkpoints = torch.load(model_file, map_location='cpu')
    sd = checkpoints["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.train()
    clipmodel = model.to(device)

    model_file = './model_pt/29_ld.torch'
    config = GPT2Config(
        vocab_sizea = 49408,
        n_layer=6,
        n_head=8,
        n_ctx=77,
    )
    gptmodel = GPT2LMHeadModel(config).cuda()
    gptmodel.load_state_dict(torch.load(model_file, map_location='cpu'), strict=False)
    gptmodel.train()
    gptmodel = gptmodel.to(device)

    for name, param in clipmodel.named_parameters():
        for i in range(11):
            s1 = "visual.transformer.resblocks." + str(i) + "."
            if s1 in name:
                param.requires_grad = False
            s2 = "visual2.transformer.resblocks." + str(i) + "."
            if s2 in name:
                param.requires_grad = False
            s3 = "transformer.resblocks." + str(i) + "."
            if s3 in name:
                param.requires_grad = False

    classmodel = ClassModel(args.input_dim, args.hidden_dim, args.output_dim)
    classmodel.train()
    classmodel = classmodel.to(device)

    train(args, logger, train_dataloader, clipmodel, gptmodel, classmodel)
    logger.close()

'''
python train.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD --logger_comment tsbir_SFSD
python train.py --dataset FScocoDataset --dataset_root_path ~/datasets/fscoco --output_dim 80 --logger_comment tsbir_fscoco
python train.py --dataset SketchycocoDataset --dataset_root_path ~/datasets/SketchyCOCO --output_dim 80 --logger_comment tsbir_sketchycoco
'''