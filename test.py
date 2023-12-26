#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:20:09
# @Author: zzm
import json
from tqdm import tqdm
import numpy as np
import os

import torch
from sklearn.neighbors import NearestNeighbors

from code.dataset import get_loader
from code.models import convert_weights, CLIP
from code.config import get_parser

def save_result(args, retrival_result):
    with open("/home/zzm/projects/MyTSBIR/runs/retrival_result.json", "w") as f:
        json.dump(retrival_result, f)

def test(args, test_dataloader, clipmodel):
    with torch.no_grad():
        recall = 0
        img_feats = []
        fused_feats = [] 
        sketch_ids = []
        retrival_result = {}
        for batch in tqdm(test_dataloader):
            sketch_id, image, sketch, txt, _, _, _, = batch
            image, sketch, txt = image.cuda(), sketch.cuda(), txt.cuda()
            image_feature, fused_feature = clipmodel(image, txt, sketch)
            img_feats.extend(image_feature.cpu().detach().numpy())
            fused_feats.extend(fused_feature.cpu().detach().numpy())
            sketch_ids.extend(list(sketch_id))

        sketch_ids = np.array(sketch_ids)
        img_feats = np.stack(img_feats)
        fused_feats = np.stack(fused_feats)
        nbrs = NearestNeighbors(n_neighbors=args.Top_K, algorithm='brute', metric='cosine').fit(img_feats)
        distances, indices = nbrs.kneighbors(fused_feats)
        for index,indice in enumerate(indices):
            if index in indice:
                recall += 1
            retrival_result[sketch_ids[index]] = list(sketch_ids[indice])
        print(round(recall / len(img_feats), 4))
        save_result(args, retrival_result)

'''
python test.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD-open --resume ./runs/Dec04_08-12-22_dp3090tsbir_SFSD_all_train_text/latest_checkpoint.pth
python test.py --dataset FScocoDataset --dataset_root_path ~/datasets/fscoco --resume ./runs/Dec05_00-38-16_dp3090tsbir_fscoco_alltexts/latest_checkpoint.pth
python test.py --dataset SketchycocoDataset --dataset_root_path ~/datasets/SketchyCOCO --resume ./runs/Dec05_15-23-17_dp3090tsbir_sketchycoco_textall/latest_checkpoint.pth
python test.py --dataset SketchycocoLFDataset --dataset_root_path ~/datasets/SketchyCOCO-lf --resume ./runs/Dec08_12-29-07_dp3090tsbir_sketchycocolf/latest_checkpoint.pth
'''
if __name__ == '__main__':
    parser = get_parser(split='test')
    args = parser.parse_args()

    test_dataloader = get_loader(args, 'test')
    model_config_file = './model_pt/ViT-B-16.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    clipmodel = model.to(device)
    convert_weights(clipmodel)
    test(args, test_dataloader, clipmodel)