import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image

class ConCapDataset:
    def __init__(
        self, ann_file, root, transforms=None, extra_args=None,
    ):
        self._image_root = root
        self._transforms = transforms
        with open(ann_file, 'r') as fin:
            self.metadata = json.load(fin)

    def __getitem__(self, idx):
        fname = self.metadata[idx]['fname']
        anno = self.metadata[idx]['caption']
        img = Image.open(os.path.join(self._image_root, fname)).convert('RGB')
        if self._transforms is not None:
            img, _ = self._transforms(img, None)
        return img, anno, idx

    def get_img_info(self, index):
        return self.metadata[index]

    def __len__(self):
        return len(self.metadata)