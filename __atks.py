import torch
import random
from __defs import Krum

### ATTACKS
class IPM():
    def __init__(self, fl, scale = 0.1):
        self.scale = scale
    def __call__(self, b_updates):
        return - self.scale * torch.stack(b_updates).mean(dim=0)
    
class MinMax():
    def __init__(self, fl):
        pass
    def __call__(self, b_updates):
        b_updates = torch.stack(b_updates)
        mu = b_updates.mean(dim=0)
        sig = b_updates.std(dim=0)
        threshold = torch.cdist(b_updates, b_updates, p=2).max()

        l, h = 0, 5
        while abs(h - l) > 0.01:
            z = (l + h) / 2
            m_grad = torch.stack([mu - z * sig])
            loss = torch.cdist(m_grad, b_updates, p=2).max()
            if loss < threshold:
                l = z
            else:
                h = z

        return mu - z * sig
class Fang():
    def __init__(self, fl):
        self.num_clients = fl.num_clients
        self.num_byz = fl.num_byz
        self.krum_fn = Krum(fl, return_index=True)
        
    def __call__(self, b_updates):
        stop_threshold = 1.0e-5
        est_direction = torch.sign(torch.mean(torch.stack(b_updates), dim=0))
        simulation_updates = torch.stack(b_updates + [torch.zeros_like(b_updates[0]) for _ in range(self.num_byz)])
        
        assert self.num_byz > 1, "FangAttack requires more than 1 attacker"
        
        lambda_value = 1.0
        while True:
            simulation_updates[self.num_clients - self.num_byz:self.num_clients] = \
                lambda_value * est_direction
            krum_idx = self.krum_fn(simulation_updates)
            if krum_idx < (self.num_clients - self.num_byz) or lambda_value <= stop_threshold:
                break
            lambda_value *= 0.5

        return lambda_value * est_direction
    
class Mimic():
    def __init__(self, fl):
        pass
        
    def __call__(self, b_updates):
        return random.choice(b_updates)
        
    
class LabelFlip():
    def __init__(self, fl):
        self.num_classes = fl.num_classes

    def __call__(self, images, labels):
        return images, self.num_classes - 1 - labels

class SignFlip():
    def __init__(self, fl):
        pass

class GaussRandom():
    def __init__(self, fl, std=20.0):
        self.device = fl.device
        self.std = std
        
    def __call__(self, noise_shape):
        return torch.normal(0, self.std, size=noise_shape).to(self.device)


__all__ = ['IPM', 'MinMax', 'Fang', 'Mimic', 'LabelFlip', 'SignFlip', 'GaussRandom']
