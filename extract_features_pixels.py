import argparse
import json
import glob
from PIL import Image
import pandas as pd
import numpy as np
import os
import os.path as osp
import torch
import scipy.io as sio
from utils.renderer import Renderer
from tqdm import tqdm
from torchvision import transforms
from models import SMPL, HMR
import config
from constants import *


COLUMNS = ["ImageName", "View", "Posture", "Identity", "InputTransform",
           "Region", "Activations"]
ACTIVATIONS_FILE_NAME = "activations.pkl"
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_RES, IMG_RES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_NORM_MEAN,
                         std=IMG_NORM_STD)
])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_transform', default="crop", choices=["crop", "raw"],
                        help="How to transform the input")
    return parser.parse_args()

def center_crop(im, res):
    width, height = im.size   # Get dimensions
    new_width, new_height = res

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def split_stim_name(stim_name):
    ident = int(stim_name[0:2])
    posture = int(stim_name[2:4])
    view = int(stim_name[4:6])
    return ident, posture, view

def feature_extraction(input_img_dir, add_info=None,
                       input_transform="crop", crop_res=None):

    img_files = glob.glob(osp.join(input_img_dir, "*"))
    activation_data = []
    if add_info is None:
        add_info = {}

    for img_file in tqdm(img_files):
        # img file is, for example, 010101.bmp
        img = Image.open(img_file)

        if input_transform == "raw":
            pass
        elif input_transform == "crop":
            assert crop_res is not None
            img = center_crop(img, res=crop_res)
        else:
            raise Exception("Not implemented.")

        img_name = osp.basename(img_file).split(".")[0]
        ident, posture, view = split_stim_name(img_name)

        # loop through regions
        out = TRANSFORM(img).numpy().flatten()
        activation_data.append(
            [img_file, view, posture, ident, input_transform, "Pixels", out]
        )

    df = pd.DataFrame(activation_data, columns=COLUMNS)
    for key, value in add_info.items():
        df[key] = value
    return df

def save_df(df, base_info, out_dir):
    for key, value in base_info.items():
        df[key] = value
    os.makedirs(out_dir, exist_ok=True)
    out_pth = osp.join(out_dir, ACTIVATIONS_FILE_NAME)
    df.to_pickle(out_pth, protocol=4)
    print(f"Saved to {out_pth}")

def main():
    """
    This will save the feature/neural activation data as a .pkl file
    to each relevant folder.
    """
    args = get_arguments()
    out_folder = "data/activations"
    img_dir = "/home/hy348/scratch60/hakan/datasets/monkey/stimuli/original"
    crop_res = (230,230)

    # information (i.e. column) to add to the final dataframe
    base_info = {"Source" : "Pixels"}
    add_info = {}

    add_info["Name"] = "Pixels"
    df = feature_extraction(img_dir, crop_res=crop_res,
                            input_transform=args.input_transform,
                            add_info=add_info)
    out_dir = "data/activations/pixels"
    save_df(df, base_info, out_dir)

if __name__ == "__main__":
    main()
