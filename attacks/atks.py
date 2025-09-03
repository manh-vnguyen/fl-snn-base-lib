import torch
from ..defenses import Krum

### ATTACKS
class IPM():
    def __init__(self, fl, scale = 0.1):
        self.scale = scale
    def __call__(self, b_updates):
        return - self.scale * torch.stack(b_updates).mean(dim=0)

class LIE():
    def __init__(self, fl, z_max = 1.5):
        self.z_max = z_max
    def __call__(self, updates):
        b_grads = torch.stack(updates)
        mu = b_grads.mean(dim=0)
        std = b_grads.std(dim=0)
        return mu - std * self.z_max

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

class MinSum():
    def __init__(self, fl):
        pass

    def __call__(self, b_grads):
        b_grads = torch.stack(b_grads)
        mu = b_grads.mean(dim=0)
        sig = b_grads.std(dim=0)
        threshold = torch.cdist(b_grads, b_grads, p=2).sum(dim=0).max()

        l, h = 0, 5
        while abs(h - l) > 0.01:
            z = (l + h) / 2
            m_grad = torch.stack([mu - z * sig])
            loss = torch.cdist(m_grad, b_grads, p=2).sum()
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

class GaussRandom():
    def __init__(self, fl, std=20.0):
        self.device = fl.device
        self.std = std
        
    def __call__(self, noise_shape):
        return torch.normal(0, self.std, size=noise_shape).to(self.device)

class LabelFlip():
    def __init__(self, fl):
        pass
    
    def __call__(self, updates):
        # LabelFlip attack doesn't modify gradients, it modifies labels during training
        # This is handled in the training loop, not here
        return updates

class SignFlip():
    def __init__(self, fl):
        pass
    
    def __call__(self, updates):
        # SignFlip attack flips the sign of gradients
        # This is handled in the training loop, not here
        return updates


__all__ = ['IPM', 'MinMax', 'Fang', 'GaussRandom', 'LIE', 'MinSum', 'LabelFlip', 'SignFlip']
