import torch
import numpy as np
from copy import deepcopy
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
import random


### AGGREGATORS
class Mean():
    def __init__(self, fl):
        self.device = fl.device

    def __call__(self, updates):
        return torch.stack(updates).mean(dim=0)

class DnC():
    def __init__(self, fl, n_iters=10, sub_dim=10000, fliter_frac=1.0):
        self.n_iters = n_iters
        self.sub_dim = sub_dim
        self.fliter_frac = fliter_frac
        self.num_byz = fl.num_byz
        self.attack_fn = fl.attack_fn

    def __call__(self, updates):
        updates = torch.stack(updates)
        d = len(updates[0])

        b_ids = []
        for i in range(self.n_iters):
            indices = torch.randint(0, d, (self.sub_dim,)).unique()
            # indices = torch.randperm(d)[: self.sub_dim] # this line is too inefficient
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            try:
                v = torch.linalg.svd(centered_update, full_matrices=False, driver='gesvd')[2][0, :]
                s = np.array(
                    [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
                )

                good = s.argsort()[
                    : len(updates) - int(self.fliter_frac * self.num_byz)
                ]
                b_ids.append(good)
            except:
                print(f"Failed SVD {self.attack_fn}")

        intersection_set = set(b_ids[0])

        for lst in b_ids[1:]:
            intersection_set.intersection_update(lst)

        b_ids = list(intersection_set)
        if len(b_ids) > 0:
            agg_grad = updates[b_ids, :].mean(dim=0)
        else:
            print(f"Failed DnC {self.attack_fn}")
            agg_grad = updates.mean(dim=0)
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

class RFA:
    def __init__(self, fl, num_iters=3, epsilon=1.0e-6, tol=1.0e-5):
        self.num_iters = num_iters  # Maximum number of iterations
        self.epsilon = epsilon  # Avoid division by zero
        self.tol = tol  # Convergence threshold
        self.device = fl.device

    def __call__(self, updates):
        updates = torch.stack(updates)

        v = torch.mean(updates, dim=0)

        for _ in range(self.num_iters):
            # Compute weights with safeguard against division by zero
            betas = 1.0 / torch.maximum(torch.norm(updates - v, p=2, dim=1), torch.tensor(self.epsilon))

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
    def __init__(self, fl, lower_bound=0.01, upper_bound=3.0, selection_fraction=0.1, clustering="KMeans"):
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
        updates = torch.sign(randomized_weights)
        sign_type = {"pos": 1, "zero": 0, "neg": -1}

        def sign_feat(sign_type):
            sign_f = (updates == sign_type).sum(dim=1, dtype=torch.float32) / num_selected
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

        sign_cluster.fit(sign_features)
        labels = sign_cluster.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 4. select the cluster with the majority of benign clients
        benign_label = np.argmax([np.sum(labels == i)
                                 for i in range(n_cluster)])
        benign_idx = [int(idx) for idx in np.argwhere(labels == benign_label)]

        return benign_idx
    
    def __call__(self, updates):
        updates = torch.stack(updates)
        # 1. filtering based on the norm of the client weights
        S1_benign_idx, median_norm, client_norms = self.norm_filtering(updates)
        
        # 2. clustering based on the sign of the client weights
        S2_benign_idx = self.sign_clustering(updates)
        
        # Find intersection of both filtering methods
        benign_idx = list(set(S1_benign_idx).intersection(S2_benign_idx))

        if len(benign_idx) == 0:
            print("Failed SignGuard, fallback to mean")
            return updates.mean(dim=0)

        # 3. clip the benign gradients by median of norms
        updates_clipped_norm = torch.clamp(
            client_norms[benign_idx], min=0, max=median_norm)
        benign_clipped = (
            updates[benign_idx] / client_norms[benign_idx].reshape(-1, 1)) * updates_clipped_norm.reshape(-1, 1)        

        return benign_clipped.mean(dim=0)

class NormClipping:
    def __init__(self, fl, norm_threshold=3, weakDP=False, noise_mean=0, noise_std=0.002, epsilon=1e-6):
        self.epsilon = epsilon
        self.norm_threshold = norm_threshold
        self.weakDP = weakDP
        if self.weakDP:
            self.noise_mean = noise_mean
            self.noise_std = noise_std
        self.device = fl.device

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
    
__all__ = ['Mean', 'DnC', 'Krum', 'RFA', 'CenterClipping', 'SignGuard', 'NormClipping']