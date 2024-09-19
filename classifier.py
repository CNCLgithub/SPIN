"""MNIST backbone image classifier example.

To run: python backbone_image_classifier.py --trainer.max_epochs=50

"""

import os.path as osp
import signal

import torch
import torch.nn as nn
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo, Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from datasets import MixedDataset, BaseDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from models import HMR
from utils import TrainOptions
import logging


logger = logging.getLogger("lightning.pytorch.core")


class LitClassifier(LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(self, backbone: str = "resnet50", learning_rate: float = 0.0001,
                 pretrained: bool = True, ief_iters: int = 0, n_classes: int = 20, n_fcs: int = 2):
        super().__init__()
        self.save_hyperparameters()
        self.backbone, fc_in_features = \
            HMR.select_backbone(backbone, pretrained)

        # define fully connected layers
        if self.hparams.ief_iters > 0:
            # IEF is turned on
            self.fc1 = nn.Linear(fc_in_features + n_classes, 1024)
        else:
            self.fc1 = nn.Linear(fc_in_features, 1024)
        self.drop1 = nn.Dropout()

        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        # output layer
        self.out = nn.Linear(1024, n_classes)

        # to initialize IEF loop
        self.register_buffer('init_vec', torch.zeros((1, n_classes)))

    def forward(self, x):
        # use forward for inference/predictions
        xf = self.backbone(x)
        self.features = self.backbone.save_output.outputs
        xf = torch.flatten(xf, 1)

        if self.hparams.ief_iters > 0:
            init_vec = self.init_vec.expand(x.shape[0], -1)
            pred_vec = init_vec
            for i in range(self.hparams.ief_iters):
                xc = torch.cat([xf, pred_vec], 1)
                xc = self.features[f"FC1_{i}"] = self.features[f"FC1"] = self.fc1(xc)
                xc = self.drop1(xc)
                if self.hparams.n_fcs == 2:
                    xc = self.features[f"FC2_{i}"] = self.features[f"FC2"] = self.fc2(xc)
                    xc = self.drop2(xc)
                outvec = self.features[f"OUTVEC_{i}"] = self.out(xc)
                pred_vec = self.features[f"PREDVEC_{i}"] = outvec + pred_vec
        else:
            xc = self.features["FC1_0"] = self.features[f"FC1"] = self.fc1(xf)
            xc = self.drop1(xc)
            if self.hparams.n_fcs == 2:
                xc = self.features["FC2_0"] = self.features[f"FC2"] = self.fc2(xc)
                xc = self.drop2(xc)
            pred_vec = self.features["PREDVEC"] = self.out(xc)
        return pred_vec

    def training_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    #https://github.com/Lightning-AI/pytorch-lightning/issues/4690#issuecomment-731152036
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # if hparams are changed on checkpoint loading, ignore parameters with different shapes
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


class MyDataModule(LightningDataModule):
    def __init__(self, options, batch_size: int = 64, num_workers: int = 16, monkey: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        ds_kwargs = dict({"is_train": True, "is_spin": False})
        if not monkey:
            self.train_ds = MixedDataset(options, ignore_3d=options.ignore_3d,
                                         **ds_kwargs)
            val_ds = [BaseDataset(None, ds, 0,
                                  use_augmentation=False, split_train=False,
                                  **ds_kwargs) for ds in self.train_ds.ds_list]
            self.val_ds = ConcatDataset(val_ds)
            n_classes = [ds.n_classes for ds in self.train_ds.datasets]
            n_classes_val = [ds.n_classes for ds in val_ds]
            n_classes.extend(n_classes_val)
            assert len(set(list(n_classes))) == 1
            self.n_classes = n_classes[0]
        else:
            self.train_ds = BaseDataset(options, "monkey", 0, ignore_3d=True,
                                       **ds_kwargs)
            self.val_ds = BaseDataset(options, "monkey", 0, ignore_3d=True,
                                     use_augmentation=False, split_train=False,
                                     **ds_kwargs)
            assert self.train_ds.n_classes == self.val_ds.n_classes
            self.n_classes = self.train_ds.n_classes


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    save_last=True,
    every_n_epochs=1,
    monitor="valid_loss",
    mode="min",
    filename="{epoch:02d}-{val_loss:.5f}",
)

def cli_main():
    log_dir = "lightning_logs"
    options = TrainOptions().parse_args()
    exp_name = f"{options.backbone}_ief:{options.ief_iters}_fcs:{options.n_fcs}"

    NUM_DEVICES = 2
    STRATEGY = "ddp_find_unused_parameters_true"
    MAX_EPOCHS = 75

    data = MyDataModule(options, options.batch_size,
                        num_workers = options.num_workers,
                        monkey = options.finetune_monkeys)

    if not options.finetune_monkeys:
        model = LitClassifier(n_classes=data.n_classes,
                              learning_rate = NUM_DEVICES*options.lr,
                              backbone = options.backbone,
                              ief_iters = options.ief_iters,
                              n_fcs = options.n_fcs,
                              )
    else:
        # ignore argparse and gather all information from the saved checkpoint
        NUM_DEVICES = 1
        STRATEGY = "auto"
        MAX_EPOCHS = 20
        version_str = "version_2"
        ckpt_last = torch.load(osp.join(log_dir, exp_name, version_str, "checkpoints", "last.ckpt"))
        ckpt_file_best = dict(list(ckpt_last['callbacks'].values())[0])["best_model_path"]
        exp_name += "-monkey"
        model = LitClassifier.load_from_checkpoint(ckpt_file_best, n_classes=data.n_classes)

    logger = TensorBoardLogger(log_dir, name=exp_name)
    trainer = Trainer(callbacks = [checkpoint_callback],
                      plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
                      logger=logger,
                      accelerator="gpu", devices=NUM_DEVICES, num_nodes=1, strategy=STRATEGY,
                      max_epochs=MAX_EPOCHS,
                      )
    trainer.fit(model, data)

if __name__ == "__main__":
    # cli_lightning_logo()
    cli_main()
