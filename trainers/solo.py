import torch
import torch.nn as nn
import random
import os

from .utils import test_model
from ..datasets import get_dataset
from .base import BaseTrainer


class SoloTrainer(BaseTrainer):
    def __init__(self, exp, device):
        self.exp = exp
        self.verbose = self.exp.verbose
        self.total_epochs = self.exp.total_epochs
        self.test_epochs = getattr(self.exp, 'test_epochs', None)
        self.test_freq = getattr(self.exp, 'test_freq', None)
        self.checkpoint_freq = getattr(self.exp, 'checkpoint_freq', None)
        self.perm_checkpoints = getattr(self.exp, 'perm_checkpoints', None)
        self.device = device

        if self.exp.run_path != None and not os.path.exists(self.exp.run_path):
            os.makedirs(self.exp.run_path)

        if getattr(self.exp, 'model_path', None) is None:
            self.exp.model_path = f'{self.exp.run_path}/s_model_{self.exp.exp_id}'
        if self.exp.checkpoint_freq != None and getattr(self.exp, 'exp_path', None) is None:
            self.exp.exp_path = f'{self.exp.run_path}/s_exp_{self.exp.exp_id}'
            assert not os.path.exists(self.exp.exp_path), "Experiment path already exists, give new path or keep running with exp_path"
        if getattr(self.exp, 'seed', None) is None:
            self.exp.seed = random.randint(0, 1e9)

        
        self.init_dataloader()

        self.model = self.init_model()
        if getattr(self.exp, 'checkpointed_epoch', None) is None:
            self.exp.checkpointed_epoch = 0
            self.exp.train_losses = []
            self.exp.test_accs = []
            self.exp.train_time = 0
        else:
            state_dict = self.get_checkpointed_state_dict()
            self.model.load_state_dict(state_dict)
        self.epoch = self.exp.checkpointed_epoch + 1

        self.optimizer = self.init_optimizer(self.model)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_model_state_dict(self):
        return self.model.state_dict()

    def init_dataloader(self):
        trainset, testset, _, _ = get_dataset(self.exp.dataset)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.exp.batch_size, shuffle=False, drop_last=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.exp.batch_size, shuffle=False, drop_last=True)

    def train_one_round(self):
        train_iter = iter(self.trainloader)
        total_loss, num_sample = 0, 0 
        for images, labels in train_iter:
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_sample += 1

        return total_loss / num_sample

    def test(self):
        return test_model(self.model, self.testloader, self.device)