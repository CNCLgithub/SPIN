"""
This is a script to write the BaseDataset (just the images and self-made labels)
to a ffcv-compliant .beton dataset. This will later be used to train a
self-supervised model fast.
"""
from __future__ import division

import os
from os.path import join
import torch
import torchvision.transforms.functional as F
import numpy as np
import cv2

import config
import constants
from utils.imutils import crop, flip_img
from datasets import BaseDataset
from torch.utils.data import DataLoader, ConcatDataset

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

WRITE_DIR = '/gpfs/milgram/scratch60/yildirim/hakan/datasets/SPIN_datasets'

class BaseDatasetFFCV(BaseDataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, dataset, **kwargs):
        super(BaseDatasetFFCV, self).__init__(None, dataset, 0, **kwargs)

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [constants.IMG_RES, constants.IMG_RES], rot=rot,
                       resize=False)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(float)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Process image
        img = self.rgb_processing(img,center, sc*scale, rot, flip, pn)
        img = F.to_pil_image(torch.from_numpy(img).float())

        if index % 500 == 0 and self.split_train:
            os.makedirs(join(WRITE_DIR, "ffcv_visualizations"), exist_ok=True)
            img.save(join(WRITE_DIR, "ffcv_visualizations", f"{self.dataset}_{index}.jpg"))

        if self.is_spin:
            pass
        else:
            label = torch.tensor(self.labels[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.imgname)

def main():
    finetune_monkeys = True

    ds_kwargs = dict({"is_train": True, "is_spin": False, "use_augmentation": False})
    if not finetune_monkeys:
        ds_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        train_ds = [BaseDatasetFFCV(ds, split_train=True,
                                    **ds_kwargs) for ds in ds_list]
        train_ds = ConcatDataset(train_ds)
        val_ds = [BaseDatasetFFCV(ds, split_train=False,
                                  **ds_kwargs) for ds in ds_list]
        val_ds = ConcatDataset(val_ds)
        suffix = ""
    else:
        train_ds = BaseDatasetFFCV("monkey", split_train=True,
                                    **ds_kwargs)
        val_ds = BaseDatasetFFCV("monkey", split_train=False,
                                  **ds_kwargs)
        suffix = "-monkey"

    def make_writer(write_path):
        writer = DatasetWriter(write_path, {
            'image': RGBImageField(write_mode="smart",
                                   max_resolution=256,
                                   compress_probability=0.5,
                                   jpeg_quality=100),
            'label': IntField(),
        }, num_workers=16)
        return writer

    writer = make_writer(join(WRITE_DIR, f"train{suffix}.beton"))
    writer.from_indexed_dataset(train_ds)
    writer = make_writer(join(WRITE_DIR, f"val{suffix}.beton"))
    writer.from_indexed_dataset(val_ds)

if __name__ == "__main__":
    main()
