import torch
import torch.nn as nn
import time

from __datasets import *
from __models import *
from __atks_defs import *
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
    def __init__(self, args):
        self.verbose = args.verbose
        if args.fl_momentum == 'local':
            self.init_actors, self.train_one_round = local_momentum_classes_and_functions()
        elif args.fl_momentum == 'global':
            self.init_actors, self.train_one_round = global_momentum_classes_and_functions()
        else:
            raise "Wrong FL momentum string"
        self.num_clients = args.num_clients
        self.num_byz = args.num_byz
        self.n_benign_clients = args.num_clients - args.num_byz
        model = init_model(args)
        trainset, testset = dataset(args.dataset, args.download_dataset)
        trainsets = IIDPartitioner(args.num_clients, args.batch_size).split_dataset(trainset)
        optimizer = torch.optim.SGD(model.parameters(), **(args.optimizer or {}))

        self.server, self.clients = self.init_actors(args, model, optimizer, trainsets, testset)
        self.attack_fn = globals()[args.attack['type']](self, **args.attack['params']) \
            if args.attack['type'] != None else None
        self.agg_fn = globals()[args.aggregator['type']](self, **args.aggregator['params'])
        self.total_epochs = args.total_epochs
    
    def train(self):
        train_losses = []
        for r in range(self.total_epochs):
            if self.verbose:
                start_time = time.time()
            train_loss = self.train_one_round(self)
            if self.verbose:
                print(f"Round: {r}, loss: {train_loss}, time: {time.time() - start_time}")
            train_losses.append(train_loss)
        return train_losses

    def test(self):
        return self.server.test()

if __name__ == '__main__':
    class Args:
        def __init__(self, model):
            self.total_epochs = 20
            self.verbose = True
            self.fl_momentum = 'local'
            self.download_dataset = False
            self.device = 'cuda:7'
            self.dataset = 'CIFAR10'
            self.batch_size = 32
            self.num_clients = 20
            self.num_byz = 0
            self.aggregator = {'type': 'Avg', 'params': {}}
            self.attack = {'type': None, 'params': {}}
            self.model = model
            if self.model.startswith('snn'):
                self.optimizer = {
                    'lr': 0.02,
                    'momentum': 0.95,
                    'weight_decay': 0.0001,
                }
                self.snn_hyperparams = {
                    'threshold': 1.0,
                    'leak': 0.99,
                    'timesteps': 9
                }
            elif self.model.startswith('ann'):
                self.model = 'ann_vgg9'
                self.optimizer = {
                    'lr': 0.01,
                    'momentum': 0.9,
                    'weight_decay': 0.0005,
                }
            
        def __repr__(self):
            if self.model.startswith('ann'):
                return f"{self.model, self.fl_momentum, self.optimizer}, {self.num_clients}:{self.num_byz}"
            elif self.model.startswith('snn'):
                return f"{self.model, self.fl_momentum, self.optimizer}, {self.num_clients}:{self.num_byz}, {self.snn_hyperparams}"
    
