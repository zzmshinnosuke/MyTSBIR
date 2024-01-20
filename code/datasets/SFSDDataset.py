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
import re

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from code.models import _transform, tokenize

MAX_LENGTH = 77
input_resolution = 224

class SFSDDataset(Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        self.split = split
        self.root_path = config.dataset_root_path
        self.images_path = os.path.join(self.root_path, "images")
        self.sketchImg_path = os.path.join(self.root_path, "sketchImgs")
        self.sketch_path = os.path.join(self.root_path, "sketches")
        self._transform = _transform(input_resolution, is_train=False)
        self.files = list()
        self.tokenizer = GPT2Tokenizer.from_pretrained("./model_pt/gpt2")
        self.load_files_path()
        self.load_categories()

    def load_categories(self):
        file = os.path.join(self.root_path, "categories_info.json")
        with open(file, 'r') as fp:
            self.categories_info = json.load(fp)

    def load_files_path(self):
        assert self.split in ['train', 'test', 'traintest'], 'unknown split {}'.format(self.split)

        captionpath = os.path.join(self.root_path, self.split+'.json')
        with open(captionpath, "r") as f:
            try:
                self.all_captions = json.load(f)
            except json.decoder.JSONDecodeError:
                print("don't have "+ captionpath)
        if self.split == "train":
            keys_image_id = self.all_captions.keys()
            for ki in keys_image_id:
                for caption in self.all_captions[ki]["captions"]:
                    self.files.append((ki, caption))
        elif self.split == "test": 
            keys_image_id = self.all_captions.keys()
            for ki in keys_image_id:
                self.files.append((ki, self.all_captions[ki]["captions"][0]))
        assert len(self.files)>0, 'no sketch json file find in {}'.format(self.root_path)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sketch_id, caption = self.files[index][0], self.files[index][1]
        caption = self.pre_caption(caption)
        with open(os.path.join(self.sketch_path, sketch_id+'.json'), 'r') as fp:
            item = json.load(fp)

        image_path = os.path.join(self.images_path, sketch_id+'.jpg')
        image = Image.open(image_path)
        sketchImg = Image.fromarray(self.json2image(item))
        image_tran = self._transform(image)
        sketch_tran = self._transform(sketchImg)
        
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

        return sketch_id, image_tran, sketch_tran, txt, cate , tokens, masks
    
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
    
    def pre_caption(self, caption, max_words=30):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")

        return caption

        