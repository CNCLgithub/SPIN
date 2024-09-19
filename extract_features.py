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
    parser.add_argument('--glob_dir', required=True, type=str,
                        help="""Specify glob for model directories
                        in which to search .pt files""")
    parser.add_argument('--input_transform', default="crop", choices=["crop", "raw"],
                        help="How to transform the input")
    parser.add_argument('--save_mesh', action="store_true",
                        help="Save inferred mesh")
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

def make_spin_overlay(smpl_model, features, img, get_mesh=False):
    pred_cam_key = sorted([key for key in list(features.keys()) if "PREDCAM" in key])[-1]
    pred_camera = features[pred_cam_key]
    renderer = Renderer(focal_length=FOCAL_LENGTH,
                        img_res=IMG_RES,
                        faces=smpl_model.faces)
    transl_z = (2 * FOCAL_LENGTH /
                (IMG_RES * pred_camera[:, 0] + 1e-9))
    camera_translation = torch.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      transl_z], dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = features["VERTICES"].squeeze(0).cpu().numpy()
    pred_vertices_norot = features["VERTICES_NR"].squeeze(0).cpu().numpy()
    img_array = np.array(img.resize((IMG_RES,IMG_RES)))/255

    # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img_array)*255
    overlay = Image.fromarray(img_shape.astype('uint8'), 'RGB')
    # grab the mesh
    mesh = renderer.get_mesh() if get_mesh else None
    white_array = np.ones([IMG_RES,IMG_RES,3],dtype=np.uint8)
    img_shape = renderer(pred_vertices_norot, camera_translation,
                         white_array)*255
    canonical = Image.fromarray(img_shape.astype('uint8'), 'RGB')

    return overlay, canonical, mesh

def feature_extraction(model, smpl_model, input_img_dir, device, img_out_dir=None, add_info=None,
                       input_transform="crop", crop_res=None, is_spin=False, save_mesh=False):

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
            pred_pose, pred_rotmat, pred_shape, pred_cam = model(TRANSFORM(img).unsqueeze(0).to(device))
            g_orient = torch.eye(3).unsqueeze(0).unsqueeze(0).to(device)
            vertices_norot = smpl_model(betas=pred_shape,
                                        body_pose=pred_rotmat[:, 1:],
                                        global_orient=g_orient,
                                        pose2rot=False).vertices
            smpl_output = smpl_model(betas=pred_shape,
                                     body_pose=pred_rotmat[:, 1:],
                                     global_orient=pred_rotmat[:, 0].unsqueeze(1),
                                     pose2rot=False)

        def split_pose(features):
            pose_keys = [key for key in list(features.keys()) if "PREDPOSE" in key]
            for pose_key in pose_keys:
                pred_pose = features[pose_key]
                pred_pose = pred_pose.view(-1, 3, 2) if model.rot6d else pred_pose.view(-1, 3, 3)
                gorient_key = pose_key.replace("PREDPOSE", "PREDGORIENT")
                features[gorient_key] = pred_pose[0]
                features[pose_key] = pred_pose[1:]

        split_pose(model.features)
        features = model.features
        smpl_gorient = features["SMPL_GORIENT"] = smpl_output.global_orient
        smpl_pose = features["SMPL_POSE"] = smpl_output.body_pose
        features["SMPL_GORIENT&POSE"] = torch.cat((smpl_gorient.flatten(),
                                                   smpl_pose.flatten()))
        features["SMPL_JOINTS"] = smpl_output.joints
        smpl_betas = features["SMPL_BETAS"] = smpl_output.betas
        features["SMPL_POSE&BETAS"] = torch.cat((smpl_pose.flatten(),
                                                 smpl_betas.flatten()))
        features["VERTICES"] = smpl_output.vertices
        features["VERTICES_NR"] = vertices_norot

        img_name = osp.basename(img_file).split(".")[0]
        ident, posture, view = split_stim_name(img_name)

        # loop through regions
        for region in features:
            out = features[region].cpu().numpy().flatten()
            activation_data.append(
                [img_file, view, posture, ident, input_transform, region, out]
            )

        if is_spin and img_out_dir is not None:
            overlay, canonical, mesh = make_spin_overlay(smpl_model, features, img, save_mesh)
            out_pth = osp.join(img_out_dir, f"{img_name}.png")
            overlay.save(out_pth)
            out_pth = osp.join(img_out_dir, f"{img_name}_canonical.png")
            canonical.save(out_pth)
            if save_mesh:
                mesh.export(osp.join(img_out_dir, f"{img_name}.obj"))

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
    logs_dir = "logs"
    networks_dir = "data/networks"
    out_folder = "data/activations"
    img_dir = "/home/hy348/scratch60/hakan/datasets/SPIN_datasets/monkey/stimuli/original"
    crop_res = (230,230)
    forward_kwargs = {}

    # information (i.e. column) to add to the final dataframe
    base_info = {"Source" : "SPIN"}
    keys_keep = ["backbone", "ief_iters", "rot6d", "n_fcs"]

    # search given folders recursively for .pt files
    model_dirs = glob.glob(osp.join(logs_dir, args.glob_dir))
    # spin_search_dir = osp.join(networks_dir, args.glob_dir, "**", "*.pt")
    # ckpt_files = glob.glob(spin_search_dir, recursive=True)
    add_info = {}
    for model_dir in model_dirs:
        out_dir = model_dir.replace(logs_dir, out_folder)
        if osp.exists(osp.join(out_dir, ACTIVATIONS_FILE_NAME)):
            print(f"Activations file exists for {model_dir}")
            # continue
        config_dir = model_dir.replace("-monkey", "") if "-monkey" in model_dir else model_dir
        config_dir = config_dir.replace("10.0", "5.0") if "-monkey" in model_dir and "10.0" in model_dir else config_dir
        with open(osp.join(config_dir, "config.json")) as f:
            mconfig = json.load(f)
        # use the latest checkpoint in the model directory
        try:
            ckpt_file = sorted(glob.glob(osp.join(model_dir, "checkpoints", "*.pt")))[-1]
        except:
            print(f"Couldn't find a ckpt file for {model_dir}")
            continue
        print(f"Working on {ckpt_file}")

        # load model
        mkwargs = {key: mconfig[key] for key in keys_keep}
        model = HMR(config.SMPL_MEAN_PARAMS, **mkwargs, pretrained=False)
        ckpt = torch.load(ckpt_file)
        add_info["epoch"] = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        smpl_model = SMPL(config.SMPL_MODEL_DIR, batch_size=1,
                          create_transl=False).to(device)

        img_out_dir = osp.join(out_dir, "overlays")
        os.makedirs(img_out_dir, exist_ok=True)

        add_info["Name"] = osp.basename(model_dir)
        for key in keys_keep:
            add_info[key] = mconfig[key]
        df = feature_extraction(model, smpl_model, img_dir, device,
                                img_out_dir=img_out_dir,
                                crop_res=crop_res, add_info=add_info,
                                is_spin=True,
                                save_mesh=args.save_mesh)
        save_df(df, base_info, out_dir)

if __name__ == "__main__":
    main()
