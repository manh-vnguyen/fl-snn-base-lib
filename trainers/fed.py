import torch

from .centralized_momentum import CentralizedMomentumTrainer
from .distributed_momentum import DistributedMomentumTrainer
from .base import BaseTrainer
from .. import IIDPartitioner
from ..defenses import get_defense_class
from ..attacks import get_attack_class


### Federated Learning
class FLTrainer(BaseTrainer, CentralizedMomentumTrainer, DistributedMomentumTrainer):
    def __init__(self, exp, device):
        super().__init__(exp, device)

        self.fl_momentum = self.exp.fl_momentum
        self.num_clients = self.exp.num_clients
        self.num_byz = self.exp.num_byz

        self.attack_fn = self.init_attack()
        self.n_clients_to_train = self.num_clients if self.exp.attack['type'] != None and self.exp.attack['type'] in ['SignFlip', 'LabelFlip'] else self.num_clients - self.num_byz

        self.agg_fn = self.init_aggregator()
        
        trainsets, testset = self.init_dataset()
        self.server, self.clients = self.init_actors(trainsets, testset, self.model, self.optimizer)
        
        if getattr(self.exp, 'checkpointed_epoch_need_tested', False) \
            and self.exp.checkpointed_epoch not in [i[0] for i in self.exp.test_accs] \
            and self.exp.checkpointed_epoch in self.exp.test_epochs:
            self.exp.test_accs.append((self.exp.checkpointed_epoch, self.test()))
            try:
                self.exp.save_json()
            except KeyboardInterrupt:
                print("Checkpointing interupted, redo...")
                self.exp.save_json()
                exit()
    
    
    def get_model_state_dict(self):
        return self.server.model.state_dict()
    
    def init_dataset(self):
        trainset, testset, _, _ = super().init_dataset()
        shuff_gen = torch.Generator().manual_seed(self.exp.seed)
        trainsets = IIDPartitioner(self.exp.num_clients, self.exp.batch_size).split_dataset(trainset, shuff_gen=shuff_gen)
        return trainsets, testset
    
    def init_attack(self):
        if self.exp.attack['type'] is None:
            return None
        attack_class = get_attack_class(self.exp.attack['type'])
        return attack_class(self, **self.exp.attack['params'])
    
    def init_aggregator(self):
        if self.exp.aggregator['type'] is None:
            return None
        defense_class = get_defense_class(self.exp.aggregator['type'])
        return defense_class(self, **self.exp.aggregator['params'])

    def simulate_attack(self):
        if self.attack_fn is None:
            pass
        elif type(self.attack_fn).__name__ == 'LabelFlip':
            pass
        elif type(self.attack_fn).__name__ == 'SignFlip':
            for i in range(self.num_clients - self.num_byz, self.num_clients):
                self.updates[i] *= -1
        elif type(self.attack_fn).__name__ == 'GaussRandom':
            m_update = self.attack_fn(self.updates[0].shape)
            self.updates += [m_update for _ in range(self.num_byz)]
        else:
            m_update = self.attack_fn(self.updates)
            self.updates += [m_update for _ in range(self.num_byz)]

    def collect_std_stats(self, updates):
        # Create the std_stats attribute if non-existence
        if getattr(self.exp, 'std_stats', None) is None:
            self.exp.std_stats = { 
                kappa: {
                    'top_k_eps': 0,
                    'non_top_k_eps': 0,
                    'count': 0
                } for kappa in [0.01, 0.05, 0.09, 0.1, 0.5, 0.9]}
        vecs = torch.stack(updates)

        mean = vecs.mean(dim=0).abs()
        eps = vecs.std(dim=0).abs()/mean

        # Loop through different sparsity levels
        for kappa in [0.01, 0.05, 0.09, 0.1, 0.5, 0.9]:
            # Calculate how many parameters to keep based on kappa
            K = kappa * len(vecs[0])
            
            # Find indices of the top K parameters by mean magnitude
            top_K_indices = torch.topk(mean, k=int(K)).indices
            
            # Create a mask for non-topK parameters (everything except top K)
            non_topK_mask = torch.ones_like(eps, dtype=torch.bool)
            non_topK_mask[top_K_indices] = False
            
            # Extract eps values for top K parameters where mean is non-zero
            # (avoiding division by zero in the eps calculation)f
            top_k_values = eps[top_K_indices][mean[top_K_indices] != 0]
            
            # Extract eps values for non-top K parameters where mean is non-zero
            non_top_k_values = eps[non_topK_mask][mean[non_topK_mask] != 0]
            
            if len(top_k_values) > 0 and len(non_top_k_values) > 0:
                self.exp.std_stats[kappa]['top_k_eps'] += top_k_values.mean().item()
                self.exp.std_stats[kappa]['non_top_k_eps'] += non_top_k_values.mean().item()
                self.exp.std_stats[kappa]['count'] += 1

    def after_attack_hook(self):
        if self.exp.collect_std_stats:
            self.collect_std_stats(self.updates)
            
    def init_actors(self, *args, **kwargs):
        if self.fl_momentum in ['local', 'distributed']:
            return DistributedMomentumTrainer.init_actors(self, *args, **kwargs)
        elif self.fl_momentum in ['global', 'centralized']:
            return CentralizedMomentumTrainer.init_actors(self, *args, **kwargs)
    
    def train_one_round(self):
        if self.fl_momentum in ['local', 'distributed']:
            return DistributedMomentumTrainer.train_one_round(self)
        elif self.fl_momentum in ['global', 'centralized']:
            return CentralizedMomentumTrainer.train_one_round(self)
    
    def test(self):
        return self.server.test()