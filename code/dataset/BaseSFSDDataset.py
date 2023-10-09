from torch.utils.data import Dataset
import json
import glob
import os
from PIL import Image, ImageDraw
import numpy as np

class BaseSFSDDataset(Dataset):
    """
    root_dir
    - sketch
        - 1.json
        - ...
    - images
        - 000000000072.jpg
        - ...
    - test_names.txt
    - train_names.txt
    """
    def __init__(self,
                 config,
                 split='train'):
        self.config = config
        self.split = split
        self.root_path = self.config.dataset_root_path
        self.images_path = os.path.join(self.root_path, "images")
        self.load_files()
    
    def load_files(self):
        assert self.split in ['train', 'test', 'traintest'], 'unknown split {}'.format(self.split)
        
        if self.split=='traintest':
            print(self.root_path)
            self.files = glob.glob(os.path.join(self.root_path, 'sketch', '*.json'))
        else:
            filename_txt = 'train_names.txt' if self.split=='train' else 'test_names.txt'
            
            filename_txt = os.path.join(self.root_path, filename_txt)
            assert os.path.exists(filename_txt), 'not find {}'.format(filename_txt)
            with open(filename_txt, 'r') as f:
                self.files = [os.path.join(self.root_path, 'sketch', line.strip()) for line in f.readlines()]
                
        assert len(self.files)>0, 'no json file find in {}'.format(self.root_path)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        with open(self.files[index], 'r') as fp:
            item = json.load(fp)
            item["filename"] = os.path.basename(self.files[index])

        imageId = str(item['reference'].split('.')[0])
        if 'captions' in item.keys():
            captions = item['captions']
        else:
            print(item["filename"])
            captions = "no caption"
        image_path = os.path.join(self.images_path, item["reference"])
        image = Image.open(image_path)
        sketch = Image.fromarray(self.json2image(item))
        
        return sketch, captions, image, imageId
    
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
        objects = info['objects']
        assert len(objects)<256,'too much object {}>=256'.format(len(objects))
        for obj in objects:
            for stroke in obj['strokes']:
                points = tuple(tuple(p) for p in stroke['points'])
                draw.line(points, fill=(0,0,0)) 
        return np.array(src_img)