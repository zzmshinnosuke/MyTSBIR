from PIL import Image
import glob
import os
import json

from torch.utils.data import Dataset

class BaseSketchycocoDataset(Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        self.split = split
        self.root_path = config.dataset_root_path
        self.images_path = os.path.join(self.root_path, "Scene", "GT")
        self.sketch_path = os.path.join(self.root_path, "Scene", "Sketch", "paper_version")
        self.files = list()
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

        captions = self.all_captions[imageId]['captions']

        image_path = file
        sketch_path = os.path.join(self.sketch_path, self.splitname, imageId + ".png")
        image = Image.open(image_path)
        sketch = Image.open(sketch_path)

        return sketch, captions, image, imageId
        