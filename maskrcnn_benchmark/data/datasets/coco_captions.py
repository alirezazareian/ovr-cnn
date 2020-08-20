import json
import numpy as np
import torch
import torchvision

class COCOCaptionsDataset(torchvision.datasets.coco.CocoCaptions):
    def __init__(
        self, ann_file, root, remove_images_without_annotations,
        transforms=None, extra_args=None,
    ):
        super(COCOCaptionsDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                if len(anno) > 0:
                    ids.append(img_id)
            self.ids = ids

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms


    def __getitem__(self, idx):
        img, anno = super(COCOCaptionsDataset, self).__getitem__(idx)

        # anno is a list of sentences. Pick one randomly.
        # TODO use a more deterministic approach, especially for validation
        anno = np.random.choice(anno)

        if self._transforms is not None:
            img, _ = self._transforms(img, None)
            
        return img, anno, idx


    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
