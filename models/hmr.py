""" Adapted from https://github.com/nkolot/SPIN"""

import torch
import torch.nn as nn
from models.vgg import vgg19
from models.resnet import resnet50
import numpy as np

import config
from utils.geometry import rot6d_to_rotmat

class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, smpl_mean_params, backbone="resnet50",
                 ief_iters=3, rot6d=True, n_fcs=2, pretrained=True):

        super(HMR, self).__init__()

        n_pose = 24 * 6 if rot6d else 24 * 9

        # if ief_iters is 0, IEF is effectively turned off
        self.ief_iters = ief_iters
        self.rot6d = rot6d
        self.n_fcs = n_fcs

        # set up the backbone
        self.backbone, fc_in_features = \
            self.select_backbone(backbone, pretrained)

        # define fully connected layers
        if self.ief_iters > 0:
            # IEF is turned on
            self.fc1 = nn.Linear(fc_in_features + n_pose + 13, 1024)
        else:
            self.fc1 = nn.Linear(fc_in_features, 1024)
        self.drop1 = nn.Dropout()

        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        # output layers
        self.outpose = nn.Linear(1024, n_pose)
        self.outshape = nn.Linear(1024, 10)
        self.outcam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.outpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.outshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.outcam.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)

        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        if not self.rot6d:
            init_pose = rot6d_to_rotmat(init_pose).flatten().unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None):

        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.backbone(x)
        self.features = self.backbone.save_output.outputs
        xf = torch.flatten(xf, 1)

        if self.ief_iters > 0:
            pred_pose = init_pose
            pred_shape = init_shape
            pred_cam = init_cam
            for i in range(self.ief_iters):
                xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
                xc = self.features[f"FC1_{i}"] = self.features[f"FC1"] = self.fc1(xc)
                xc = self.drop1(xc)
                if self.n_fcs == 2:
                    xc = self.features[f"FC2_{i}"] = self.features["FC2"] =  self.fc2(xc)
                    xc = self.drop2(xc)
                outpose = self.features[f"OUTPOSE_{i}"] = self.outpose(xc)
                outshape = self.features[f"OUTSHAPE_{i}"] = self.outshape(xc)
                outcam = self.features[f"OUTCAM_{i}"] = self.outcam(xc)
                pred_pose = self.features[f"PREDPOSE_{i}"] = outpose + pred_pose
                pred_shape = self.features[f"PREDSHAPE_{i}"] = outshape + pred_shape
                pred_cam = self.features[f"PREDCAM_{i}"] = outcam + pred_cam
        else:
            xc = self.features["FC1_0"] = self.features["FC1"] = self.fc1(xf)
            xc = self.drop1(xc)
            if self.n_fcs == 2:
                xc = self.features["FC2_0"] = self.features["FC2"] = self.fc2(xc)
                xc = self.drop2(xc)
            pred_pose = self.features[f"OUTPOSE_0"] = self.features[f"PREDPOSE_0"] = self.outpose(xc)
            pred_shape = self.features[f"OUTSHAPE_0"] = self.features[f"PREDSHAPE_0"] = self.outshape(xc)
            pred_cam = self.features[f"OUTCAM_0"] = self.features[f"PREDCAM_0"] = self.outcam(xc)

        if self.rot6d:
            pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        else:
            pred_rotmat = pred_pose.view(batch_size, 24, 3, 3)

        return pred_pose, pred_rotmat, pred_shape, pred_cam

    @staticmethod
    def select_backbone(name, pretrained):
        if name == "resnet50":
            model_backbone = resnet50(weights="IMAGENET1K_V1")
            fc_in_features = model_backbone.fc.in_features
            model_backbone.fc = nn.Identity()
        elif name == "vgg19":
            model_backbone = vgg19(weights='DEFAULT' if pretrained else None)
            fc_in_features = model_backbone.classifier[0].in_features
            model_backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError("Backbone not implemented.")

        return model_backbone, fc_in_features
