"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

DATASETS_ROOT="/gpfs/milgram/scratch60/yildirim/hakan/datasets/SPIN_datasets"
H36M_ROOT = join(DATASETS_ROOT, "h36m")
LSP_ROOT = join(DATASETS_ROOT, "lsp")
LSP_ORIGINAL_ROOT = join(DATASETS_ROOT, "lsp_original")
LSPET_ROOT = join(DATASETS_ROOT, "hr-lspet")
MPII_ROOT = join(DATASETS_ROOT, "mpii")
COCO_ROOT = join(DATASETS_ROOT, "coco")
MPI_INF_3DHP_ROOT = join(DATASETS_ROOT, "mpi_inf_3dhp")
PW3D_ROOT = join(DATASETS_ROOT, "3dpw")
UPI_S1H_ROOT = join(DATASETS_ROOT, "upi-s1h")
MONKEY_ROOT = join(DATASETS_ROOT, "monkey")

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   'monkey-test' : join(DATASET_NPZ_PATH, 'monkey_test.npz'),
                   'monkey-stimuli' : join(DATASET_NPZ_PATH, 'monkey_stimuli.npz'),
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                   'monkey' : join(DATASET_NPZ_PATH, 'monkey_train.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                   'monkey': MONKEY_ROOT,
                   'monkey-test': MONKEY_ROOT,
                   'monkey-stimuli': MONKEY_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
# STATIC_FITS_DIR = 'data/static_fits'
STATIC_FITS_DIR = 'data/spin_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
