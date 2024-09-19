"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import json
import os
import os.path as osp
import glob
import argparse
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm
import csv

import config
import constants
from models import SMPL, HMR
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.geometry import perspective_projection
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer

def keypoint_loss(pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight
    criterion_keypoints = nn.MSELoss(reduction='none')
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', required=True, help='Name of log folder to search for checkpoints')
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint or folder with network checkpoints')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp', 'monkey-test', 'monkey-stimuli'],
                    help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--last_n_ckpts', default=1, type=int, help="Evaluate the last n checkpoints in a checkpoint folder, if provided")
parser.add_argument('--overwrite', action="store_true")

def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # 2D keypoint loss metrics
    keypoint_err = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_joints = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']
    elif "monkey" in dataset_name:
        eval_joints = True

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        if "monkey" not in dataset_name:
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            gender = batch['gender'].to(device)
            gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        else:
            gt_keypoints_2d = batch['keypoints'].to(device)
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]
        
        with torch.no_grad():
            _, pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            
        # 2D keypoints evaluation
        if eval_joints:
            focal_length = constants.FOCAL_LENGTH
            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:, 1],
                                        pred_camera[:, 2],
                                        2 * focal_length / (img_res * pred_camera[:, 0] + 1e-9)],
                                        dim=-1)

            camera_center = torch.zeros(batch_size, 2, device=device)
            pred_keypoints_2d = perspective_projection(pred_joints,
                                                        rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                                        translation=pred_cam_t,
                                                        focal_length=focal_length,
                                                        camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (img_res / 2.)

            # Compute 2D reprojection loss for the keypoints
            error = keypoint_loss(pred_keypoints_2d, gt_keypoints_2d, 0.0, 1.0)
            keypoint_err[step * batch_size:step * batch_size + curr_batch_size] = error.cpu()

        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error


        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = renderer(pred_vertices, pred_camera)

        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Pose Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print()
            if eval_joints:
                print('Joints Reconstruction Error: ' + str(1000 * keypoint_err[:step * batch_size].mean()))
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    results = {}
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        results['Pose_MPJE'] = 1000 * mpjpe.mean()
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        results['Pose_rec_error'] = 1000 * recon_err.mean()
        print()
    if eval_joints:
        print("Joints reconstruction error: " + str(1000 * keypoint_err.mean()))
        results['Joints_rec_error'] = 1000 * keypoint_err.mean()
        print()
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        results['Mask_accuracy'] = accuracy / pixel_count
        print('F1: ', f1.mean())
        results['Mask_F1'] = f1.mean()
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        results['Parts_accuracy'] = parts_accuracy / parts_pixel_count
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        results['Parts_F1'] = parts_f1[[0,1,2,3,4,5,6]].mean()
        print()

    return results

def main():
    args = parser.parse_args()
    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, 0, is_train=False)
    csv_out_name = f"eval_{args.dataset}.csv"

    if args.checkpoint and osp.isfile(args.checkpoint):
        checkpoints = [args.checkpoint]
        csv_out_pth = osp.join(osp.dirname(args.checkpoint), csv_out_name)
        config_dir = osp.dirname(osp.dirname(args.checkpoint))
    elif osp.isdir(args.model_dir) and osp.isdir(osp.join(args.model_dir, "checkpoints")):
        ckpt_dir = osp.join(args.model_dir, "checkpoints")
        config_dir = osp.join(args.model_dir)
        # take the last n checkpoints in the folder
        checkpoints = sorted(glob.glob(osp.join(ckpt_dir, "*.pt")))
        checkpoints = checkpoints[-args.last_n_ckpts:]
        csv_out_pth = osp.join(ckpt_dir, csv_out_name)
    else:
        raise ValueError("Provide either a name or checkpoint file")

    if osp.isfile(csv_out_pth) and not args.overwrite:
        print(f"A file {csv_out_pth} already exists, nothing will happen.")
        return

    results = []
    for checkpoint_pth in tqdm(checkpoints):
        print(f"Working on checkpoint {checkpoint_pth}")

        # load architecture parameters from config
        config_dir = config_dir.replace("-monkey", "") if "-monkey" in config_dir else config_dir
        with open(osp.join(config_dir, "config.json")) as f:
            mconfig = json.load(f)
        keys_keep = ["backbone", "ief_iters", "rot6d", "n_fcs"]
        mkwargs = {key: mconfig[key] for key in keys_keep}

        # load model
        model = HMR(config.SMPL_MEAN_PARAMS, **mkwargs, pretrained=False)
        checkpoint = torch.load(checkpoint_pth)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        step_count = checkpoint['total_step_count']

        # Run evaluation
        row = run_evaluation(model, args.dataset, dataset, args.result_file,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.num_workers,
                    log_freq=args.log_freq)
        row['step_count'] = step_count
        row['ckpt_name'] = osp.basename(checkpoint_pth)
        row['dataset_name'] = args.dataset
        results.append(row)

    with open(csv_out_pth, 'w') as f:
        w = csv.DictWriter(f, results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved results to {csv_out_pth}")

if __name__ == '__main__':
    main()
