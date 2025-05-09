import torch
import numpy as np
import os
import cv2
from torchgeometry import angle_axis_to_rotation_matrix

import config
import constants

class FitsDict():
    """ Dictionary keeping track of the best fit per image in the training set """
    def __init__(self, options, train_dataset):
        self.options = options
        self.ds_dict = train_dataset.ds_dict
        self.fits_dict = {}
        # array used to flip SMPL pose parameters
        self.flipped_parts = torch.tensor(constants.SMPL_POSE_FLIP_PERM, dtype=torch.int64)
        # Load dictionary state
        for ds_name, ds_id in self.ds_dict.items():
            try:
                dict_file = os.path.join(options.checkpoint_dir, ds_name + '_fits.npy')
                self.fits_dict[ds_id] = torch.from_numpy(np.load(dict_file))
            except IOError:
                # Dictionary does not exist, so populate with static fits
                print(f"Dictionary for ds {ds_name} does not exist, so populate with static fits")
                dict_file = os.path.join(config.STATIC_FITS_DIR, ds_name + '_fits.npy')
                self.fits_dict[ds_id] = torch.from_numpy(np.load(dict_file))

    def save(self):
        """ Save dictionary state to disk """
        for ds_name, ds_id in self.ds_dict.items():
            dict_file = os.path.join(self.options.checkpoint_dir, ds_name + '_fits.npy')
            np.save(dict_file, self.fits_dict[ds_id].cpu().numpy())

    def __getitem__(self, x):
        """ Retrieve dictionary entries """
        ds_id, ind, rot, is_flipped = x
        batch_size = len(ds_id)
        pose = torch.zeros((batch_size, 72))
        betas = torch.zeros((batch_size, 10))
        for ds, i, n in zip(ds_id, ind, range(batch_size)):
            params = self.fits_dict[ds][i]
            pose[n, :] = params[:72]
            betas[n, :] = params[72:]
        pose = pose.clone()
        # Apply flipping and rotation
        pose = self.flip_pose(self.rotate_pose(pose, rot), is_flipped)
        betas = betas.clone()
        return pose, betas

    def __setitem__(self, x, val):
        """ Update dictionary entries """
        ds_id, ind, rot, is_flipped, update = x
        pose, betas = val
        batch_size = len(ds_id)
        # Undo flipping and rotation
        pose = self.rotate_pose(self.flip_pose(pose, is_flipped), -rot)
        params = torch.cat((pose, betas), dim=-1).cpu()
        for ds, i, n in zip(ds_id, ind, range(batch_size)):
            if update[n]:
                self.fits_dict[ds][i] = params[n]

    def flip_pose(self, pose, is_flipped):
        """flip SMPL pose parameters"""
        # is_flipped = is_flipped.byte()
        pose_f = pose.clone()
        pose_f[is_flipped, :] = pose[is_flipped][:, self.flipped_parts]
        # we also negate the second and the third dimension of the axis-angle representation
        pose_f[is_flipped, 1::3] *= -1
        pose_f[is_flipped, 2::3] *= -1
        return pose_f

    def rotate_pose(self, pose, rot):
        """Rotate SMPL pose parameters by rot degrees"""
        pose = pose.clone()
        cos = torch.cos(-np.pi * rot / 180.)
        sin = torch.sin(-np.pi * rot/ 180.)
        zeros = torch.zeros_like(cos)
        r3 = torch.zeros(cos.shape[0], 1, 3, device=cos.device)
        r3[:,0,-1] = 1
        R = torch.cat([torch.stack([cos, -sin, zeros], dim=-1).unsqueeze(1),
                       torch.stack([sin, cos, zeros], dim=-1).unsqueeze(1),
                       r3], dim=1)
        global_pose = pose[:, :3]
        global_pose_rotmat = angle_axis_to_rotation_matrix(global_pose)
        global_pose_rotmat_3b3 = global_pose_rotmat[:, :3, :3]
        global_pose_rotmat_3b3 = torch.matmul(R, global_pose_rotmat_3b3.double())
        global_pose_rotmat[:, :3, :3] = global_pose_rotmat_3b3
        global_pose_rotmat = global_pose_rotmat[:, :-1, :-1].cpu().numpy()
        global_pose_np = np.zeros((global_pose.shape[0], 3))
        for i in range(global_pose.shape[0]):
            aa, _ = cv2.Rodrigues(global_pose_rotmat[i])
            global_pose_np[i,:] = aa.squeeze()
        pose[:, :3] = torch.from_numpy(global_pose_np).to(pose.device)
        return pose
