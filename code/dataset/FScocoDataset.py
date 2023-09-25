from PIL import Image
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from code.clip import _transform, tokenize

MAX_LENGTH = 77
input_resolution = 224

class FScocoDataset(Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        self.split = split
        self.root_path = config.dataset_root_path
        self.images_path = os.path.join(self.root_path, "images")
        self.sketch_path = os.path.join(self.root_path, "raster_sketches")
        self._transform = _transform(input_resolution, is_train=False)
        self.files = list()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.load_files_path()

    def load_files_path(self):
        assert self.split in ['train', 'val', 'valid', 'test', 'trainval'], 'unknown split {}'.format(self.split)

        filename_txt = 'FScocoTrain.txt' if self.split == 'train' else 'FScocoTest.txt'
        filename_path = os.path.join(self.root_path, filename_txt)
        assert os.path.exists(filename_path), 'not find {}'.format(filename_txt)

        self.files = []
        with open(filename_path, 'r') as f:
            for line in f.readlines():
                self.files.append(line.strip())

        assert len(self.files) > 0, 'no txt file find in {}'.format(self.root_path)

        catfile_name = 'FScocoTrain_cat.json' if self.split == 'train' else 'FScocoTest_cat.json'
        catpath = os.path.join(self.root_path, catfile_name)
        with open(catpath, "r") as f:
            try:
                self.all_cats = json.load(f)
            except json.decoder.JSONDecodeError:
                print("don't have "+ catpath)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_id = self.files[index]
        caption_path = os.path.join(self.root_path, 'text', image_id + '.txt')
        with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.readline().strip()

        image_path = os.path.join(self.images_path, image_id + ".jpg")
        sketch_path = os.path.join(self.sketch_path, image_id + ".jpg")
        image = Image.open(image_path)
        sketch = Image.open(sketch_path)
        image_tran = self._transform(image)
        sketch_tran = self._transform(sketch)
        
        cate =  torch.tensor(np.array(list(self.all_cats[image_id]["cats"]))) 
        
        tokenized = self.tokenizer.encode("<|endoftext|> " + caption + " <|endoftext|>")[:MAX_LENGTH]
        masks = torch.zeros(MAX_LENGTH)
        masks[torch.arange(len(tokenized))] = 1
        tokens = torch.zeros(MAX_LENGTH).long()
        tokens[torch.arange(len(tokenized))] = torch.LongTensor(tokenized)

        txt = tokenize([str(caption)])[0]

        return image_tran, sketch_tran, txt, cate , tokens, masks
        