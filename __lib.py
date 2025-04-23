import torch
import torch.nn as nn
import time
import pickle
import random
import os

from __datasets import *
from __models import *
import __defs as defs
import __atks as atks
from __actors import *

display_time = lambda seconds: "%d:%02d:%02d" % (
    (seconds) // 3600,
    (seconds % 3600) // 60,
    seconds % 60
)

## Solo training
class SoloTrainer():
    def __init__(self, args):
        self.device = args.device
        trainset, testset = dataset(args.dataset)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        self.model = init_model(args)
        self.optimizer = torch.optim.SGD(self.model.parameters(), **(args.optimizer or {}))
        self.loss_fn = nn.CrossEntropyLoss()
        

    def train(self, epochs):
        for r in range(epochs):
            train_loss, train_total = 0, 0 
            train_iter = iter(self.trainloader)
            for images, labels in train_iter:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.loss_fn(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_total += 1
            print(f"Round {r}, loss: {train_loss/train_total}")

    def test(self):
        with torch.no_grad():
            acc, total = 0, 0 
            for images, labels in iter(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                acc += torch.sum(torch.argmax(output, dim=1) == labels)
                total += len(labels)
        return (acc/total).item()


### Federated Learning
class FLTrainer():
    def __init__(self, exp, ext_model_path=None, device='cpu'):
        self.verbose = exp.verbose
        self.num_clients = exp.num_clients
        self.num_byz = exp.num_byz
        self.device = device
        self.total_epochs = exp.total_epochs
        self.test_epochs = getattr(exp, 'test_epochs', None)
        self.test_freq = getattr(exp, 'test_freq', None)
        self.checkpoint_freq = getattr(exp, 'checkpoint_freq', None)
        self.perm_checkpoints = getattr(exp, 'perm_checkpoints', None)

        if not os.path.exists(exp.run_path):
            os.makedirs(exp.run_path)

        if getattr(exp, 'model_path', None) is None:
            exp.model_path = f'{exp.run_path}/model_{exp.exp_num}'
        if getattr(exp, 'exp_path', None) is None:
            exp.exp_path = f'{exp.run_path}/exp_{exp.exp_num}'
            assert not os.path.exists(exp.exp_path), "Experiment path already exists, gives new path or keep running with exp_path"
        if getattr(exp, 'seed', None) is None:
            exp.seed = random.randint(0, 1e9)

        shuff_gen = torch.Generator().manual_seed(exp.seed)
        trainset, testset, self.num_classes = dataset(exp.dataset, exp.download_dataset)
        trainsets = IIDPartitioner(exp.num_clients, exp.batch_size).split_dataset(trainset, shuff_gen=shuff_gen)

        self.attack_fn = getattr(atks, exp.attack['type'])(self, **exp.attack['params']) if exp.attack['type'] != None else None
        self.n_clients_to_train = self.num_clients if exp.attack['type'] != None and exp.attack['type'] in ['SignFlip', 'LabelFlip'] else self.num_clients - self.num_byz

        self.agg_fn = getattr(defs, exp.aggregator['type'])(self, **exp.aggregator['params'])

        model = init_model(exp, self.device, self.num_classes)
        if getattr(exp, 'checkpointed_epoch', None) is None:
            exp.checkpointed_epoch = 0
            exp.train_losses = []
            exp.test_accs = []
            exp.train_time = 0
        else:
            if ext_model_path != None:
                state_dict = torch.load(ext_model_path)
            else:
                state_dict = torch.load(exp.model_path)
            model.load_state_dict(state_dict)

        self.epoch = exp.checkpointed_epoch + 1

        optimizer = torch.optim.SGD(model.parameters(), **(exp.optimizer or {}))

        if exp.fl_momentum == 'global':
            self.server, self.clients = init_actors_global_momentum(exp, model, optimizer, trainsets, testset, self.attack_fn, self.device)
            self.train_one_round = train_one_round_global_momentum
        else:
            raise "Wrong FL momentum string"
        
        if getattr(exp, 'checkpointed_epoch_need_tested', False) \
            and exp.checkpointed_epoch not in [i[0] for i in exp.test_accs] \
            and exp.checkpointed_epoch in exp.test_epochs:
            exp.test_accs.append((exp.checkpointed_epoch, self.test()))
            try:
                pickle.dump(exp, open(exp.exp_path, 'wb'))
            except KeyboardInterrupt:
                print("Checkpointing interupted, redo...")
                pickle.dump(exp, open(exp.exp_path, 'wb'))
                exit()
        
        self.exp = exp
    
    def test(self):
        return self.server.test()

    def checkpoint(self, train_losses, test_accs, train_time, perm_checkpoint=False):
        self.exp.train_losses += train_losses
        self.exp.test_accs += test_accs
        self.exp.train_time += train_time
        self.exp.checkpointed_epoch = self.epoch

        try:
            torch.save(self.server.model.state_dict(), self.exp.model_path)
            if perm_checkpoint:
                torch.save(self.server.model.state_dict(), f"{self.exp.model_path}_ep{self.epoch}")
            pickle.dump(self.exp, open(self.exp.exp_path, 'wb'))
        except KeyboardInterrupt:
            print("Checkpointing interupted, redo...")
            torch.save(self.server.model.state_dict(), self.exp.model_path)
            if perm_checkpoint:
                torch.save(self.server.model.state_dict(), f"{self.exp.model_path}_ep{self.epoch}")
            pickle.dump(self.exp, open(self.exp.exp_path, 'wb'))
            exit()
    
    def run(self):
        self.start_training = time.time()
        train_losses, test_accs = [], []
        while self.epoch <= self.total_epochs:
            if self.verbose:
                start_time = time.time()
            train_loss = self.train_one_round(self)
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

        if os.path.exists(self.exp.model_path):
            os.remove(self.exp.model_path)

        return train_losses, test_accs

if __name__ == '__main__':
    class ExpRecord:
        def __init__(self, model):
            self.exp_num = 11
            self.run_path = '/home/combined_everything_FL/run_data/find_eq_ann_snn'
            self.seed = 22032025
            self.total_epochs = 2000
            self.checkpoint_freq = 10
            self.test_epochs = list(range(self.total_epochs - 200, self.total_epochs + 1, 50))
            self.perm_checkpoints = []
            self.verbose = True
            self.fl_momentum = 'global'
            self.download_dataset = True
            self.dataset = 'CIFAR100'
            self.batch_size = 32
            self.num_clients = 20
            self.num_byz = 0
            self.aggregator = {'type': 'Mean', 'params': {}}
            self.attack = {'type': None, 'params': {}}
            self.model = model
            if self.model.startswith('snn'):
                self.optimizer = {
                    'lr': 0.15,
                    'momentum': 0.95,
                    'weight_decay': 0.0001,
                }
                self.snn_hyperparams = {
                    'threshold': 1.5,
                    'leak': 0.95,
                    'timesteps': 15
                }
            elif self.model.startswith('ann'):
                self.model = 'ann_vgg9'
                self.optimizer = {
                    'lr': 0.01,
                    'momentum': 0.9,
                    'weight_decay': 0.0005,
                }


    # exp = pickle.load(open('/home/combined_everything_FL/run_data/find_eq_ann_snn/exp_10', 'rb'))
    # exp = ExpRecord('snn_vgg9')
    exp = pickle.load(open('/home/combined_everything_FL/run_data/r_snn_atks_defs_surrs/exp_618', 'rb'))
    
    fl_trainer = FLTrainer(
        exp=exp,
        device='cuda:5')
    fl_trainer.run()