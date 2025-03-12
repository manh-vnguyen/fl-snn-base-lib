import torch
import numpy as np
from copy import deepcopy
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
import random


### AGGREGATORS
class Mean():
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


    def __call__(self, updates):
        updates = torch.stack(updates, dim=0)
        d = len(updates[0])

        b_ids = []
        for i in range(self.n_iters):
            indices = torch.randint(0, d, (self.sub_dim,)).unique()
            # indices = torch.randperm(d)[: sub_dim] this line is too inefficient
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array(
                [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
            )

            good = s.argsort()[
                : len(updates) - int(self.fliter_frac * self.num_byz)
            ]
            b_ids.append(good)

        intersection_set = set(b_ids[0])

        for lst in b_ids[1:]:
            intersection_set.intersection_update(lst)

        b_ids = list(intersection_set)
        agg_grad = updates[b_ids, :].mean(dim=0)
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

class RFA:
    def __init__(self, fl, num_iters=3, epsilon=1.0e-6, tol=1.0e-5):
        self.num_iters = num_iters  # Maximum number of iterations
        self.epsilon = epsilon  # Avoid division by zero
        self.tol = tol  # Convergence threshold

    def __call__(self, updates):
        updates = torch.stack(updates)

        v = torch.mean(updates, dim=0)

        for _ in range(self.num_iters):
            differences = updates - v
            norms = torch.norm(differences, p=2, dim=1)

            # Compute weights with safeguard against division by zero
            betas = 1.0 / torch.maximum(norms, torch.tensor(self.epsilon))

            # Compute new estimate of v
            v_new = torch.sum(betas[:, None] * updates, dim=0) / betas.sum()

            # Check for convergence
            if torch.norm(v_new - v, p=2) < self.tol:
                break

            v = v_new  

        return v

class CenterClipping:
    def __init__(self, fl, norm_threshold=100, num_iters=1):
        self.norm_threshold = norm_threshold
        self.num_iters = num_iters
        self.momentum = None

    def __call__(self, updates):
        # Initialize momentum with correct shape if not already initialized
        if self.momentum is None:
            self.momentum = torch.zeros_like(updates[0], dtype=torch.float32)
        
        num_updates = len(updates)
        for _ in range(self.num_iters):
            clipped_sum = torch.zeros_like(self.momentum)
            
            for update in updates:
                diff = update - self.momentum
                norm = torch.norm(diff, p=2)
                if norm > self.norm_threshold:
                    scale = self.norm_threshold / norm
                    clipped_sum += diff * scale
                else:
                    clipped_sum += diff
            
            self.momentum += clipped_sum / num_updates

        return deepcopy(self.momentum)


class SignGuard():
    def __init__(self, fl, lower_bound=0.1, upper_bound=3.0, selection_fraction=0.1, clustering="DBSCAN"):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.selection_fraction = selection_fraction
        self.clustering = clustering

    def norm_filtering(self, updates):
        client_norms = torch.linalg.norm(updates, dim=1)
        median_norm = torch.median(client_norms)
        mask = (client_norms > self.lower_bound * median_norm) & (client_norms < self.upper_bound * median_norm)
        benign_idx = torch.nonzero(mask).flatten().tolist()
        return benign_idx, median_norm, client_norms

    def sign_clustering(self, updates):
        # 1. randomized coordinate selection
        num_para = updates.shape[1]
        num_selected = int(self.selection_fraction * num_para)
        idx = random.randint(0, int((1 - self.selection_fraction) * num_para))
        
        # 2. extract positive, negative, and zero sign statistics
        randomized_weights = updates[:, idx:(idx + num_selected)]
        sign_grads = torch.sign(randomized_weights)
        sign_type = {"pos": 1, "zero": 0, "neg": -1}

        def sign_feat(sign_type):
            sign_f = (sign_grads == sign_type).sum(dim=1, dtype=torch.float32) / num_selected
            return sign_f / (sign_f.max() + 1e-8)
            
        num_clients = updates.shape[0]
        sign_features = torch.empty((num_clients, 3), dtype=torch.float32)
        sign_features[:, 0] = sign_feat(sign_type["pos"])
        sign_features[:, 1] = sign_feat(sign_type["zero"])
        sign_features[:, 2] = sign_feat(sign_type["neg"])

        # Convert to numpy for clustering algorithms
        sign_features_np = sign_features.numpy()
        
        # 3. clustering based on the sign statistics
        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(sign_features_np, quantile=0.5, n_samples=50)
            sign_cluster = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
        elif self.clustering == "DBSCAN":
            sign_cluster = DBSCAN(eps=0.05, min_samples=3)
        elif self.clustering == "KMeans":
            sign_cluster = KMeans(n_clusters=2)

        sign_cluster.fit(sign_features_np)
        labels = torch.tensor(sign_cluster.labels_)
        n_cluster = len(set(sign_cluster.labels_)) - (1 if -1 in sign_cluster.labels_ else 0)
        
        # 4. select the cluster with the majority of benign clients
        cluster_counts = [torch.sum(labels == i).item() for i in range(n_cluster)]
        benign_label = torch.argmax(torch.tensor(cluster_counts)).item()
        benign_idx = torch.nonzero(labels == benign_label).flatten().tolist()
        return benign_idx
    
    def __call__(self, updates):
        updates = torch.stack(updates)
        # 1. filtering based on the norm of the client weights
        S1_benign_idx, median_norm, client_norms = self.norm_filtering(updates)
        
        # 2. clustering based on the sign of the client weights
        S2_benign_idx = self.sign_clustering(updates)
        
        # Find intersection of both filtering methods
        benign_idx = list(set(S1_benign_idx).intersection(S2_benign_idx))

        # 3. clip the benign gradients by median of norms
        grads_clipped_norm = torch.clamp(
            client_norms[benign_idx], min=0, max=median_norm)
        benign_clipped = (
            updates[benign_idx] / client_norms[benign_idx].reshape(-1, 1)) * grads_clipped_norm.reshape(-1, 1)

        return benign_clipped.mean(dim=0)

class NormClipping:
    def __init__(self, fl, norm_threshold=3, weakDP=False, noise_mean=0, noise_std=0.002, epsilon=1e-6):
        self.epsilon = epsilon
        self.norm_threshold = norm_threshold
        self.weakDP = weakDP
        if self.weakDP:
            self.noise_mean = noise_mean
            self.noise_std = noise_std

    def __call__(self, updates):
        updates = torch.stack(updates)
        # norm clipping
        updates = updates * torch.minimum(torch.tensor(1.0), self.norm_threshold / (torch.norm(updates, dim=1) + self.epsilon)).reshape(-1, 1)
        # add noise to clients' updates
        if self.weakDP:
            # add gaussian noise to the vector, z~N(0, sigma^2 * I)
            updates += np.random.normal(self.noise_mean, self.noise_std,
                                    updates.shape).astype(np.float32)
        return updates.mean(dim=0)

### ATTACKS
class LIE():
    def __init__(self, fl, z_max = 1.5):
        self.z_max = z_max
    def __call__(self, b_updates):
        b_updates = torch.stack(b_updates)
        mu = b_updates.mean(dim=0)
        std = b_updates.std(dim=0)
        return mu - std * self.z_max

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
    
class MinSum():
    def __init__(self, fl):
        pass

    def __call__(self, b_updates):
        b_updates = torch.stack(b_updates)
        mu = b_updates.mean(dim=0)
        sig = b_updates.std(dim=0)
        threshold = torch.cdist(b_updates, b_updates, p=2).sum(dim=0).max()

        l, h = 0, 5
        while abs(h - l) > 0.01:
            z = (l + h) / 2
            m_grad = torch.stack([mu - z * sig])
            loss = torch.cdist(m_grad, b_updates, p=2).sum()
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

if __name__ == '__main__':
    torch.manual_seed(0)
    updates = [torch.randn(30) for _ in range(5)]  # Example updates (5 clients, 10-dimensional)
    print(torch.stack(updates).sum())
    
    print('SignGuard', SignGuard(None)(updates))
    print('NormClipping', NormClipping(None)(updates))
    print('CenterClipping', CenterClipping(None)(updates))
    print('RFA', RFA(None)(updates))
    print('Median', Median(None)(updates))
    print('TrimmedMean', TrimmedMean(None)(updates))
    print('Mean', Mean(None)(updates))
    print('TopK', TopK(None)(updates))
    
