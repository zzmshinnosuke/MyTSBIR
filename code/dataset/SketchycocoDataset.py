from PIL import Image
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

class SketchycocoDataset(Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        self.split = split
        self.root_path = config.dataset_root_path
        self.images_path = os.path.join(self.root_path, "Scene", "GT")
        self.sketch_path = os.path.join(self.root_path, "Scene", "Sketch", "paper_version")
        self._transform = _transform(input_resolution, is_train=False)
        self.files = list()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.load_files_path()

    def load_files_path(self):
        assert self.split in ['train', 'test', 'val'], 'unknown split {}'.format(self.split)

        if self.split == "train":
            self.splitname = 'trainInTrain'
        elif self.split == "val":
            self.splitname = 'val'
        elif self.split == "test":
            self.splitname = 'valInTrain'
        self.files = glob.glob(os.path.join(self.images_path, self.splitname, '*.png'))
        assert len(self.files)>0, 'no sketch json file find in {}'.format(self.root_path)

        captionpath = os.path.join(self.root_path, self.splitname+'.json')
        with open(captionpath, "r") as f:
            try:
                self.all_captions = json.load(f)
            except json.decoder.JSONDecodeError:
                print("don't have "+ captionpath)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        imageId = os.path.basename(file).split(".")[0]

        caption = self.all_captions[imageId]['captions'][0]

        image_path = file
        sketch_path = os.path.join(self.sketch_path, self.splitname, imageId + ".png")
        image = Image.open(image_path)
        sketch = Image.open(sketch_path)
        image_tran = self._transform(image)
        sketch_tran = self._transform(sketch)
        
        cate = torch.tensor(np.array(list(self.all_captions[imageId]['cats'])))
        
        tokenized = self.tokenizer.encode("<|endoftext|> " + caption + " <|endoftext|>")[:MAX_LENGTH]
        masks = torch.zeros(MAX_LENGTH)
        masks[torch.arange(len(tokenized))] = 1
        tokens = torch.zeros(MAX_LENGTH).long()
        tokens[torch.arange(len(tokenized))] = torch.LongTensor(tokenized)

        txt = tokenize([str(caption)])[0]

        return image_tran, sketch_tran, txt, cate , tokens, masks
        