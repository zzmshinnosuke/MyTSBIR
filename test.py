#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:20:09
# @Author: zzm
import json
from tqdm import tqdm
import numpy as np

import torch
from sklearn.neighbors import NearestNeighbors

from code.dataset import get_loader
from code.clip import convert_weights, CLIP, tokenize
from code.config import get_parser

def test(test_dataloader, clipmodel):
    with torch.no_grad():
        recall = 0
        img_feats = []
        fused_feats = [] 
        for batch in tqdm(test_dataloader):
            image, sketch, txt, _, _, _, = batch
            image, sketch, txt = image.cuda(), sketch.cuda(), txt.cuda()
            image_feature, fused_feature = clipmodel(image, txt, sketch)
            img_feats.extend(image_feature.cpu().detach().numpy())
            fused_feats.extend(fused_feature.cpu().detach().numpy())

        img_feats = np.stack(img_feats)
        fused_feats = np.stack(fused_feats)
        nbrs = NearestNeighbors(n_neighbors=Top_K, algorithm='brute', metric='cosine').fit(img_feats)
        distances, indices = nbrs.kneighbors(fused_feats)
        for index,indice in enumerate(indices):
            if index in indice:
                recall += 1
        print(round(recall / len(img_feats), 4))

'''
python test.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD --resume ./runs/Sep19_17-32-00_cu02tsbir_SFSD/latest_checkpoint.pth
python test.py --dataset FScocoDataset --dataset_root_path ~/datasets/fscoco --resume ./runs/Sep19_17-33-48_cu02tsbir_fscoco/latest_checkpoint.pth
python test.py --dataset SketchycocoDataset --dataset_root_path ~/datasets/SketchyCOCO --resume ./runs/Sep19_20-02-14_cu02tsbir_sketchycoco/latest_checkpoint.pth
'''
if __name__ == '__main__':
    parser = get_parser(split='test')
    args = parser.parse_args()

    test_dataloader = get_loader(args, 'test')

    Top_K = 1

    model_config_file = './code/training/model_configs/ViT-B-16.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    clipmodel = model.to(device)
    convert_weights(clipmodel)
    test(test_dataloader, clipmodel)