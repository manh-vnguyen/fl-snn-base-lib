import torch
import torch.nn as nn
import time
import pickle

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
    def __init__(self, exp=None, exp_path=None):
        assert exp_path != None ^ exp != None, "Exactly one variable should be input (exp, exp_path)"
        if exp_path:
            exp = pickle.load(open(exp_path, 'rb'))
        self.verbose = exp.verbose
        self.num_clients = exp.num_clients
        self.num_byz = exp.num_byz
        self.device = exp.device
        self.total_epochs = exp.total_epochs
        trainset, testset, self.num_classes = dataset(exp.dataset, exp.download_dataset)
        
        if getattr(exp, 'model_path') is None:
            exp.model_path = f'{exp.run_path}/model_{exp.exp_num}'
        if getattr(exp, 'exp_path') is None:
            exp.exp_path = f'{exp.run_path}/exp_{exp.exp_num}'
        if getattr(exp, 'seed') is None:
            exp.seed = random.randint(0, 1e9)
        shuff_gen = torch.Generator().manual_seed(exp.seed)

        trainsets = IIDPartitioner(exp.num_clients, exp.batch_size).split_dataset(trainset, shuff_gen=shuff_gen)
        optimizer = torch.optim.SGD(model.parameters(), **(exp.optimizer or {}))

        self.attack_fn = globals()[exp.attack['type']](self, **exp.attack['params']) if exp.attack['type'] != None else None
        self.n_clients_to_train = self.num_clients if exp.attack['type'] != None and exp.attack['type'] in ['SignFlip', 'LabelFlip'] else self.num_clients - self.num_byz

        self.agg_fn = globals()[exp.aggregator['type']](self, **exp.aggregator['params'])

        if getattr(exp, 'checkpointed_epoch') is None:
            exp.checkpointed_epoch = -1
            model = init_model(exp, self.num_classes)
            exp.train_losses = []
            exp.test_accs = []
            self.epoch = self.exp.checkpointed_epoch + 1
        else:
            torch.load(exp.model_path)

        if exp.fl_momentum == 'global':
            self.server, self.clients = init_actors_global_momentum(exp, model, optimizer, trainsets, testset, self.attack_fn, shuff_gen, exp.checkpointed_epoch)
            self.train_one_round = train_one_round_global_momentum
        else:
            raise "Wrong FL momentum string"
        
        self.exp = exp
    
    def test(self):
        return self.server.test()

    def checkpoint(self, train_losses, test_accs):
        self.exp.train_losses += train_losses
        self.exp.test_accs += test_accs
        torch.save(self.server.model, self.exp.model_path)
        pickle.dump(self.exp, open(self.exp.exp_path, 'wb'))
    
    def train(self):
        train_losses, test_accs = [], []
        while self.epoch < self.total_epochs:
            if self.verbose:
                start_time = time.time()
            train_loss = self.train_one_round(self)
            train_losses.append(train_loss)
            self.epoch += 1
            if self.epoch % self.test_freq == 0:
                acc = self.test()
                test_accs.append((self.epoch, acc))
                if self.verbose:
                    print(f"Epoch: {self.epoch}, loss: {train_loss}, acc: {acc}, time: {time.time() - start_time}")
            else:
                if self.verbose:
                    print(f"Epoch: {self.epoch}, loss: {train_loss}, time: {time.time() - start_time}")
            if self.epoch % self.checkpoint_freq == 0:
                self.checkpoint(train_losses, test_accs)
                train_losses, test_accs = [], []
            

    

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
            self.num_byz = 8
            self.aggregator = {'type': 'Mean', 'params': {}}
            self.attack = {'type': 'LIE', 'params': { 'z_max': 1.0}}
            self.model = model
            if self.model.startswith('snn'):
                self.optimizer = {
                    'lr': 0.02,
                    'momentum': 0.95,
                    'weight_decay': 0.0001,
                }
                self.snn_hyperparams = {
                    'threshold': 1.5,
                    'leak': 0.95,
                    'timesteps': 25
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
            
    FLTrainer(Args('ann_vgg9')).train()
    
