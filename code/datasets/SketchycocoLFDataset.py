from PIL import Image
import numpy as np
import os
import json
import re

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from code.models import _transform, tokenize

MAX_LENGTH = 77
input_resolution = 224

class SketchycocoLFDataset(Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        self.split = split
        self.root_path = config.dataset_root_path
        self.images_path = os.path.join(self.root_path)
        self.sketch_path = os.path.join(self.root_path)
        self._transform = _transform(input_resolution, is_train=False)
        self.files = list()
        self.tokenizer = GPT2Tokenizer.from_pretrained("./model_pt/gpt2")
        self.load_files_path()

    def load_files_path(self):
        assert self.split in ['train', 'test', 'val'], 'unknown split {}'.format(self.split)
 
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
        imageId, caption = self.files[index][0], self.files[index][1]
        caption = self.pre_caption(caption)

        image_path = os.path.join(self.images_path, self.split, 'image', imageId + ".png")
        sketch_path = os.path.join(self.sketch_path, self.split, 'sketch', imageId + ".png")
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

        return imageId, image_tran, sketch_tran, txt, cate , tokens, masks
    
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
        