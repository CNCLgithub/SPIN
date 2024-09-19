import argparse
import glob
from PIL import Image
import pandas as pd
import numpy as np
import os
import os.path as osp
import torch
from tqdm import tqdm
from torchvision import transforms
from classifier import LitClassifier
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
    parser.add_argument('--glob_dir', required=True, type=str,
                        help="""Specify glob for model directories
                        in which to search .pt files""")
    parser.add_argument('--input_transform', default="crop", choices=["crop", "raw"],
                        help="How to transform the input")
    parser.add_argument('--save_mesh', action="store_true",
                        help="Save inferred mesh")
    parser.add_argument('--skip', action="store_true",
                        help="Skip if activations file exists")
    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite if activations file exists")
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

def feature_extraction(model, input_img_dir, device, add_info=None,
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

        with torch.no_grad():
            model(TRANSFORM(img).unsqueeze(0).to(device))

        features = model.features

        img_name = osp.basename(img_file).split(".")[0]
        ident, posture, view = split_stim_name(img_name)

        # loop through regions
        for region in features:
            out = features[region].cpu().numpy().flatten()
            activation_data.append(
                [img_file, view, posture, ident, input_transform, region, out]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logs_dir = "lightning_logs"
    out_folder = "data/activations/classification"
    img_dir = "/home/hy348/scratch60/hakan/datasets/SPIN_datasets/monkey/stimuli/original"
    crop_res = (230,230)
    forward_kwargs = {}

    # information (i.e. column) to add to the final dataframe
    base_info = {"Source" : "Classification"}

    # search given folders recursively for .pt files
    model_dirs = glob.glob(osp.join(logs_dir, args.glob_dir))
    # spin_search_dir = osp.join(networks_dir, args.glob_dir, "**", "*.pt")
    # ckpt_files = glob.glob(spin_search_dir, recursive=True)
    add_info = {}
    for model_dir in model_dirs:

        # directory for output
        out_dir = model_dir.replace(logs_dir, out_folder)
        if osp.exists(osp.join(out_dir, ACTIVATIONS_FILE_NAME)):
            message = f"Activations file exists for {model_dir}."
            if args.skip: print(message+" Skipping.."); continue
            if not args.overwrite: raise Exception(message)

        # use the latest checkpoint to find the best in terms of validation loss
        ckpt = torch.load(osp.join(model_dir, "checkpoints", "last.ckpt"))
        ckpt_best_pth = dict(list(ckpt['callbacks'].values())[0])["best_model_path"]
        ckpt = torch.load(ckpt_best_pth)
        print(f"Working on {ckpt_best_pth}")

        # load model
        model = LitClassifier.load_from_checkpoint(ckpt_best_pth)
        model.to(device)
        model.eval()

        add_info["Name"] = osp.normpath(model_dir).split(osp.sep)[1]
        add_info["epoch"] = ckpt['epoch']
        for key in ["ief_iters", "n_fcs", "backbone"]:
            add_info[key] = getattr(model.hparams, key)

        # extract features
        df = feature_extraction(model, img_dir, device,
                                crop_res=crop_res, add_info=add_info
                                )
        save_df(df, base_info, out_dir)

if __name__ == "__main__":
    main()
