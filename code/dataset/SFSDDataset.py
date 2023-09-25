#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 10:50:39
# @Author: zzm

from PIL import Image
import PIL.ImageDraw as ImageDraw
import numpy as np
import glob
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from code.clip import _transform, tokenize

MAX_LENGTH = 77
input_resolution = 224

class SFSDDataset(Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        self.split = split
        self.root_path = config.dataset_root_path
        self.images_path = os.path.join(self.root_path, "images")
        self.sketch_path = os.path.join(self.root_path, "sketchImg")
        self._transform = _transform(input_resolution, is_train=False)
        self.files = list()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.load_files_path()
        self.load_categories()

    def load_categories(self):
        file = os.path.join(self.root_path, "categories_info.json")
        with open(file, 'r') as fp:
            self.categories_info = json.load(fp)

    def load_files_path(self):
        assert self.split in ['train', 'test', 'traintest'], 'unknown split {}'.format(self.split)

        if self.split == 'traintest':
            self.files = glob.glob(os.path.join(self.root_path, 'sketch', '*.json'))
        else: 
            filename_txt = 'train_names.txt' if self.split == 'train' else 'test_names.txt'
            filename_path = os.path.join(self.root_path, filename_txt)
            assert os.path.exists(filename_path), 'not find {}'.format(filename_path)
            with open(filename_path, 'r') as f:
                self.files=[os.path.join(self.root_path, 'sketch', line.strip()) for line in f.readlines()]
        assert len(self.files)>0, 'no sketch json file find in {}'.format(self.root_path)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], 'r') as fp:
            item = json.load(fp)
            item["filename"] = os.path.basename(self.files[index])

        imageId = str(item['reference'].split('.')[0])
        if 'captions' in item.keys():
            caption = item['captions'][0]
        else:
            print(item["filename"])
            caption = "test"
        image_path = os.path.join(self.images_path, item["reference"])
        # sketch_path = os.path.join(self.sketch_path, imageId + ".jpeg")
        image = Image.open(image_path)
        # sketch = Image.open(sketch_path)
        sketch = Image.fromarray(self.json2image(item))
        image_tran = self._transform(image)
        sketch_tran = self._transform(sketch)
        
        categories = dict.fromkeys(self.categories_info, 0)
        for obj in item['objects']:
            categories[obj['category']] = 1
        cate = torch.tensor(np.array(list(categories.values())))
        
        tokenized = self.tokenizer.encode("<|endoftext|> " + caption + " <|endoftext|>")[:MAX_LENGTH]
        masks = torch.zeros(MAX_LENGTH)
        masks[torch.arange(len(tokenized))] = 1
        tokens = torch.zeros(MAX_LENGTH).long()
        tokens[torch.arange(len(tokenized))] = torch.LongTensor(tokenized)

        txt = tokenize([str(caption)])[0]

        return image_tran, sketch_tran, txt, cate , tokens, masks
    
    def json2image(self, info):
        """
        info.keys(): ['filename', 'resolution', 'captions', 'scene', 'objects']
        objects[0].keys(): ['name', 'category', 'strokes', 'integrity', 
                            'similarity', 'color', 'id', 'direction', 'quality']
        strokes[0].keys(): ['color', 'thickness', 'id', 'points']
        """
        # width,height
        width, height = info['resolution']
        src_img = Image.new("RGB", (width,height), (255,255,255))
        draw = ImageDraw.Draw(src_img)       
        objects=info['objects']
        assert len(objects)<256,'too much object {}>=256'.format(len(objects))
        for obj in objects:
            for stroke in obj['strokes']:
                points=tuple(tuple(p) for p in stroke['points'])
                draw.line(points, fill=(0,0,0)) 
        return np.array(src_img)

        