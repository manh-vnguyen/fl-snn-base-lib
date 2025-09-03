import torch
import torch.nn as nn
import random
import os

from .utils import test_model
from ..datasets import get_dataset
from .base import BaseTrainer


class SoloTrainer(BaseTrainer):
    def __init__(self, exp, device):
        super().__init__(exp, device)
        self.init_dataloader()

    def get_model_state_dict(self):
        return self.model.state_dict()

    def init_dataloader(self):
        trainset, testset, _, _ = super().init_dataset()
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.exp.batch_size, shuffle=False, drop_last=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.exp.batch_size, shuffle=False, drop_last=True)

    def train_one_round(self):
        train_iter = iter(self.trainloader)
        total_loss, num_sample = 0, 0 
        for images, labels in train_iter:
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            loss = torch.nn.functional.cross_entropy(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_sample += 1

        return total_loss / num_sample

    def test(self):
        return test_model(self.model, self.testloader, self.device)