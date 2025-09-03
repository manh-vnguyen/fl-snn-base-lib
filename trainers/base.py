import torch
import time
import random
import os
import logging

from .. import init_model, get_dataset
from ..experiments import ExpFed, ExpSolo

class BaseTrainer():
    def __init__(self, exp, device):
        self.exp = exp
        self.verbose = self.exp.verbose
        self.total_epochs = self.exp.total_epochs
        self.test_epochs = getattr(self.exp, 'test_epochs', None)
        self.test_freq = getattr(self.exp, 'test_freq', None)
        self.checkpoint_freq = getattr(self.exp, 'checkpoint_freq', None)
        self.perm_checkpoints = getattr(self.exp, 'perm_checkpoints', None)
        self.device = device

        if not os.path.exists(self.exp.run_path):
            os.makedirs(self.exp.run_path)

        if type(self.exp) is ExpFed:
            prefix = 'f'
        elif type(self.exp) is ExpSolo:
            prefix = 's'
        else:
            raise Exception("exp type is not right")

        if getattr(self.exp, 'model_path', None) is None:
            self.exp.model_path = f'{self.exp.run_path}/{prefix}_{self.exp.exp_id}_model.pth'
        if self.exp.checkpoint_freq != None and getattr(self.exp, 'exp_path', None) is None:
            self.exp.exp_path = f'{self.exp.run_path}/{prefix}_{self.exp.exp_id}_exp.json'
            assert not os.path.exists(self.exp.exp_path), "Experiment path already exists, give new path or keep running with exp_path"
        if getattr(self.exp, 'seed', None) is None:
            self.exp.seed = random.randint(0, 1e9)

        # Set up logger
        self.logger = logging.getLogger(f"Trainer_{self.exp.exp_id}")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if self.exp.run_path is not None:
            log_file = os.path.join(self.exp.run_path, f"{prefix}_{self.exp.exp_id}.log")
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        # Prevent propagation to root logger (avoids duplicate messages)
        self.logger.propagate = False

        self.model = self.init_model()
        if getattr(self.exp, 'checkpointed_epoch', None) is None:
            self.exp.checkpointed_epoch = 0
            self.exp.train_losses = []
            self.exp.test_accs = []
            self.exp.train_time = 0
            self.logger.info("Initialized new experiment state.")
        else:
            state_dict = self.get_checkpointed_state_dict()
            self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded checkpointed state dict from {self.exp.model_path}.")
        self.epoch = self.exp.checkpointed_epoch + 1

        self.optimizer = self.init_optimizer(self.model)
        self.lr_scheduler = self.init_lr_scheduler(self.optimizer)


    def get_checkpointed_state_dict(self):
        return torch.load(self.exp.model_path)

    def init_dataset(self):
        return get_dataset(**self.exp.dataset)

    def init_model(self):
        return init_model(self.exp, self.device)

    def init_optimizer(self, model):
        return getattr(torch.optim, self.exp.optimizer['type'])(model.parameters(), **self.exp.optimizer['params'])
    
    def init_lr_scheduler(self, optimizer):
        if self.exp.lr_scheduler is None:
            return None
        return getattr(torch.optim.lr_scheduler, self.exp.lr_scheduler['type'])(optimizer, **self.exp.lr_scheduler['params'])
    
    def train_one_round(self):
        pass
    
    def test(self):
        pass

    def get_checkpointed_state_dict(self):
        return torch.load(self.exp.model_path)

    def get_model_state_dict(self):
        pass

    def checkpoint(self, train_losses, test_accs, train_time, perm_checkpoint=False):
        self.exp.train_losses += train_losses
        self.exp.test_accs += test_accs
        self.exp.train_time += train_time
        self.exp.checkpointed_epoch = self.epoch

        try:
            torch.save(self.get_model_state_dict(), self.exp.model_path)
            if perm_checkpoint:
                torch.save(self.get_model_state_dict(), f"{self.exp.model_path}_ep{self.epoch}")
            self.exp.save_json()
            # if self.verbose:
            #     self.logger.info(f"Checkpointed at epoch {self.epoch}.")
        except KeyboardInterrupt:
            self.logger.warning("Checkpointing interrupted by user, retrying...")
            torch.save(self.get_model_state_dict(), self.exp.model_path)
            if perm_checkpoint:
                torch.save(self.get_model_state_dict(), f"{self.exp.model_path}_ep{self.epoch}")
            self.exp.save_json()
            self.logger.info(f"Checkpointed at epoch {self.epoch} after interruption.")
            exit()
    
    def run(self):
        self.logger.info("Starting training run.")
        self.start_training = time.time()
        train_losses, test_accs = [], []
        while self.epoch <= self.total_epochs:
            if self.verbose:
                start_time = time.time()
            train_loss = self.train_one_round()
            train_losses.append(train_loss)

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(train_loss)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (self.test_freq != None and self.epoch % self.test_freq == 0) \
                or (self.test_epochs != None and self.epoch in self.test_epochs):
                acc = self.test()
                test_accs.append((self.epoch, acc))
                if self.verbose:
                    self.logger.info(f"Epoch: {self.epoch}, loss: {train_loss}, acc: {acc}, time: {time.time() - start_time}")
            else:
                if self.verbose:
                    self.logger.info(f"Epoch: {self.epoch}, loss: {train_loss}, time: {time.time() - start_time}")
            if (self.checkpoint_freq != None and self.epoch % self.checkpoint_freq == 0) \
                or (self.perm_checkpoints != None and self.epoch in self.perm_checkpoints) \
                or self.epoch == self.total_epochs:
                self.checkpoint(train_losses, test_accs, time.time() - self.start_training, (self.perm_checkpoints != None and self.epoch in self.perm_checkpoints))
                train_losses, test_accs = [], []
                self.start_training = time.time()

            self.epoch += 1

        if not self.exp.checkpoint_retain_last_model:
            if os.path.exists(self.exp.model_path):
                os.remove(self.exp.model_path)
                self.logger.info(f"Removed last model file at {self.exp.model_path} as per configuration.")

        self.logger.info("Training run complete.")
        return train_losses, test_accs
    
__all__ = ['BaseTrainer']