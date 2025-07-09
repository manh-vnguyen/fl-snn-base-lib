import torch
import time
import random
import os

from .global_momentum import GlobalMomentumTrainer
from .local_momentum import LocalMomentumTrainer
from .. import get_dataset, IIDPartitioner, init_model
from ..defenses import get_defense_class
from ..attacks import get_attack_class

class BaseTrainer():
    def get_checkpointed_state_dict(self):
        return torch.load(self.exp.model_path)

    def init_dataset(self):
        pass

    def init_model(self):
        return init_model(self.exp, self.device)

    
    def init_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), **(self.exp.optimizer or {}))
    
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
        except KeyboardInterrupt:
            print("Checkpointing interupted, redo...")
            torch.save(self.get_model_state_dict(), self.exp.model_path)
            if perm_checkpoint:
                torch.save(self.get_model_state_dict(), f"{self.exp.model_path}_ep{self.epoch}")
            self.exp.save_json()
            exit()
    
    def run(self):
        self.start_training = time.time()
        train_losses, test_accs = [], []
        while self.epoch <= self.total_epochs:
            if self.verbose:
                start_time = time.time()
            train_loss = self.train_one_round()
            train_losses.append(train_loss)
            if (self.test_freq != None and self.epoch % self.test_freq == 0) \
                or (self.test_epochs != None and self.epoch in self.test_epochs):
                acc = self.test()
                test_accs.append((self.epoch, acc))
                if self.verbose:
                    print(f"Epoch: {self.epoch}, loss: {train_loss}, acc: {acc}, time: {time.time() - start_time}")
            else:
                if self.verbose:
                    print(f"Epoch: {self.epoch}, loss: {train_loss}, time: {time.time() - start_time}")
            if (self.checkpoint_freq != None and self.epoch % self.checkpoint_freq == 0) \
                or (self.perm_checkpoints != None and self.epoch in self.perm_checkpoints):
                self.checkpoint(train_losses, test_accs, time.time() - self.start_training, (self.perm_checkpoints != None and self.epoch in self.perm_checkpoints))
                train_losses, test_accs = [], []
                self.start_training = time.time()

            self.epoch += 1

        if not self.exp.checkpoint_retain_last_model:
            if os.path.exists(self.exp.model_path):
                os.remove(self.exp.model_path)

        return train_losses, test_accs
    
__all__ = ['BaseTrainer']