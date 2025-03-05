import torch
import numpy as np

### AGGREGATORS
class Avg():
    def __init__(self, fl):
        pass
    def __call__(self, grads):
        return torch.stack(grads).mean(dim=0)

class TopK():
    def __init__(self, fl, kappa = 0.1):
        self.kappa = kappa
        self.K = None
        
        
    def __call__(self, grads):
        if self.K is None:
            self.K = int(self.kappa * len(grads[0]))
        count = torch.zeros_like(grads[0])
        total = torch.zeros_like(grads[0])
        for input_tensor in grads:
            threshold = torch.topk(torch.abs(input_tensor), k=self.K)[0][-1]
            mask = torch.abs(input_tensor) >= threshold
            total += input_tensor * mask
            count += mask
        return total / count.clamp(min=1)
        
class DnC():
    def __init__(self, fl, n_iters=10, sub_dim=1000, fliter_frac=1.0):
        self.n_iters = n_iters
        self.sub_dim = sub_dim
        self.fliter_frac = fliter_frac
        self.num_byz = fl.num_byz


    def __call__(self, b_grads):
        grads = torch.stack(b_grads, dim=0)
        d = len(grads[0])

        b_ids = []
        for i in range(self.n_iters):
            indices = torch.randint(0, d, (self.sub_dim,)).unique()
            # indices = torch.randperm(d)[: sub_dim] this line is too inefficient
            sub_grads = grads[:, indices]
            mu = sub_grads.mean(dim=0)
            centered_update = sub_grads - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array(
                [(torch.dot(update - mu, v) ** 2).item() for update in sub_grads]
            )

            good = s.argsort()[
                : len(grads) - int(self.fliter_frac * self.num_byz)
            ]
            b_ids.append(good)

        intersection_set = set(b_ids[0])

        for lst in b_ids[1:]:
            intersection_set.intersection_update(lst)

        b_ids = list(intersection_set)
        agg_grad = grads[b_ids, :].mean(dim=0)
        return agg_grad

class Krum():
    def __init__(self, fl, return_index=False):
        self.num_clients = fl.num_clients
        self.num_byz = fl.num_byz
        self.return_index = return_index

    def __call__(self, updates):
        assert 2 * self.num_byz + 2 < self.num_clients, f"num_byzantine should meet 2f+2 < n, got 2*{self.num_byz}+2 >= {self.num_clients}."
        
        if not isinstance(updates, torch.Tensor):
            updates = torch.stack(updates)
        
        distances = torch.cdist(updates, updates, p=2)
        
        # For each client, compute the sum of distances to the closest (n-f-1) clients
        scores = torch.zeros(self.num_clients)
        for i in range(self.num_clients):
            client_distances = distances[i]
            # Sort distances and sum the smallest (n-f-1) values
            scores[i] = torch.sum(torch.sort(client_distances)[0][1:self.num_clients-self.num_byz])
        
        max_score_index = torch.argmax(scores).item()
        
        if not self.return_index:
            return updates[max_score_index]
        else:
            return max_score_index

class Median():
    def __init__(self, fl):
        pass

    def __call__(self, updates):
        return torch.median(torch.stack(updates), dim=0).values
    
class TrimmedMean():
    def __init__(self, fl, filter_frac=0.1):
        self.filter_frac = filter_frac

    def __call__(self, updates):
        updates = torch.stack(updates)
        num_excluded = int(self.filter_frac * len(updates))
        sorted_updates, _ = torch.sort(updates, dim=0)
        smallest_excluded = sorted_updates[:num_excluded]
        biggest_excluded = sorted_updates[-num_excluded:]
        
        weights = updates.sum(dim=0) - smallest_excluded.sum(dim=0) - biggest_excluded.sum(dim=0)
        weights /= (len(updates) - 2 * num_excluded)
        
        return weights

### ATTACKS
class LIE():
    def __init__(self, fl, z_max = 1.5):
        self.z_max = z_max
    def __call__(self, b_grads):
        b_grads = torch.stack(b_grads)
        mu = b_grads.mean(dim=0)
        std = b_grads.std(dim=0)
        return mu - std * self.z_max

class IPM():
    def __init__(self, fl, scale = 0.1):
        self.scale = scale
    def __call__(self, b_grads):
        return - self.scale * torch.stack(b_grads).mean(dim=0)
    
class MinMax():
    def __init__(self, fl):
        pass
    def __call__(self, b_grads):
        b_grads = torch.stack(b_grads)
        mu = b_grads.mean(dim=0)
        sig = b_grads.std(dim=0)
        threshold = torch.cdist(b_grads, b_grads, p=2).max()

        l, h = 0, 5
        while abs(h - l) > 0.01:
            z = (l + h) / 2
            m_grad = torch.stack([mu - z * sig])
            loss = torch.cdist(m_grad, b_grads, p=2).max()
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
        
    def __call__(self, b_grads):
        stop_threshold = 1.0e-5
        est_direction = torch.sign(torch.mean(torch.stack(b_grads), dim=0))
        simulation_updates = torch.stack(b_grads + [torch.zeros_like(b_grads[0]) for _ in range(self.num_byz)])
        
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

