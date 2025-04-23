import time
from itertools import product
import pickle
import argparse
import os

BASE_PATH = '/home/combined_everything_FL/run_data'

class ExpRecord:
    def __init__(self, model):
        self.seed = 22032025
        self.total_epochs = 2000
        self.checkpoint_freq = 10
        self.test_epochs = [1900, 1950, 2000]
        self.perm_checkpoints = [1900, 2000]
        self.verbose = True
        self.fl_momentum = 'global'
        self.download_dataset = True
        self.dataset = 'CIFAR100'
        self.batch_size = 32
        self.num_clients = 20
        self.num_byz = 8
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
            self.optimizer = {
                'lr': 0.01, 
                'momentum': 0.95, 
                'weight_decay': 0.0001
            }

def compile_and_save_exp(config):
    dataset, model, fl_momentum, surrogate, attack, aggregator, exp_num, run_id = config
    
    exp = ExpRecord(model)
    exp.dataset = dataset
    exp.exp_num = exp_num
    exp.run_path = f"{BASE_PATH}/r_{run_id}"
    if not os.path.exists(exp.run_path):
        os.makedirs(exp.run_path)
    exp.exp_path = f'{exp.run_path}/exp_{exp.exp_num}'

    exp.fl_momentum = fl_momentum
    exp.attack = attack
    exp.aggregator = aggregator
    if surrogate != None:
        exp.snn_hyperparams['surrogate'] = surrogate
    
    assert not os.path.exists(exp.exp_path), "Error, trying to override"

    pickle.dump(exp, open(exp.exp_path, 'wb'))

def gen_exps(run_id):
    all_experiments = []

    contexts = [
        ('CIFAR10', 'snn_vgg9', 'global'),
        ('CIFAR100', 'snn_vgg9', 'global'),
        ('MNIST', 'snn_fc', 'global'),
        ('FMNIST', 'snn_fc', 'global'),
    ]

    ann_contexts = [
        ('CIFAR10', 'ann_vgg9', 'global'),
        ('CIFAR100', 'ann_vgg9', 'global'),
        ('MNIST', 'ann_fc', 'global'),
        ('FMNIST', 'ann_fc', 'global'),
    ]

    attacks = [
        {'type': None, 'params': {}},
        {'type': 'Fang', 'params': {}},
        {'type': 'MinMax', 'params': {}},
        {'type': 'IPM', 'params': { 'scale': 1.1}},
        {'type': 'Mimic', 'params': {}},
        {'type': 'SignFlip', 'params': {}},
        {'type': 'LabelFlip', 'params': {}},
        {'type': 'GaussRandom', 'params': {'std': 1.0}},
    ]

    aggregators = [
        {'type': 'Mean', 'params': {}},
        {'type': 'NormClipping', 'params': {}},
        {'type': 'DnC', 'params': {}},
        {'type': 'RFA', 'params': { 'num_iters': 20}},
        {'type': 'CenterClipping', 'params': {}},
        {'type': 'SignGuard', 'params': {}},
    ]

    surrogates = [
        {'type': 'TriangleSurr', 'params': {}},
        {'type': 'FastSigmoidSurr', 'params': {}},
        {'type': 'GaussianSurr', 'params': {}},
        {'type': 'QuadraticSurr', 'params': {}},
        {'type': 'RectangleSurr', 'params': {}},
    ]
    
    count = 0
    for attack, aggregator, context, surrogate in product(attacks, aggregators, contexts, surrogates):
        if attack['type'] == None and aggregator['type'] != 'Mean':
            continue
        all_experiments.append((*context, surrogate, attack, aggregator, count, run_id))
        count += 1

    for attack, aggregator, context, surrogate in product(attacks, aggregators, ann_contexts, [None]):
        if attack['type'] == None and aggregator['type'] != 'Mean':
            continue
        all_experiments.append((*context, surrogate, attack, aggregator, count, run_id))
        count += 1

    for config in all_experiments:
        compile_and_save_exp(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID')
    parsed_args = parser.parse_args()
    if parsed_args.run_id != None:
        gen_exps(parsed_args.run_id)
    else:
        gen_exps('snn_exps')