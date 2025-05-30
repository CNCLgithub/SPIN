from __future__ import division
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.utils.tensorboard import SummaryWriter

from ffcv_dataset import make_ffcv_loader
from utils import CheckpointDataLoader, CheckpointSaver


class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    # self.models_dict[model].load_state_dict(checkpoint[model], strict=False)
                    self.models_dict[model].load_state_dict(checkpoint[model])
                    print('Checkpoint loaded')

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            def checkpoint():
                self.loss_summary(losses)
                self.img_summary(batch, output)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count)
                tqdm.write('Checkpoint saved')

            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              )):
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    output, losses = self.train_step(batch)
                    if np.isnan(losses["loss"]):
                        print("Encountered NaN loss.")
                    self.step_count += 1

                    # log at very first step to make sure everything works
                    if self.epoch_count == 0 and self.step_count == 1:
                        checkpoint()

                    # # Tensorboard logging every summary_steps steps
                    # if self.step_count % self.options.summary_steps == 0:
                    #     self.loss_summary(losses)
                    # # Save checkpoint every checkpoint_steps steps
                    # if self.step_count % self.options.checkpoint_steps == 0:
                    #     self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count,
                    #                                self.options.model_backbone, self.options.forward_iters, self.options.no_ief)
                    #     tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.finalize()
                    # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count,
                    #                             self.options.model_backbone, self.options.forward_iters, self.options.no_ief)
                    # tqdm.write('Checkpoint saved')
                    checkpoint()
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            # if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)

            # we're done with the epoch, log and save checkpoint
            # self.loss_summary(losses)
            # self.img_summary(labels, output)
            # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count,
            #                            self.options.model_backbone, self.options.forward_iters, self.options.no_ief)
            checkpoint()
            self.finalize()
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def loss_summary(self):
        raise NotImplementedError('You need to provide a _loss_summary method')

    def img_summary(self, input_batch):
        raise NotImplementedError('You need to provide a _img_summary method')

    def test(self):
        pass
