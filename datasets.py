import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, Subset
from typing import List

def get_dataset(name, z_norm = False, download=False):
    data_dir = f"/tmp/data"
    transform_list = [transforms.ToTensor()]
    if z_norm:
        if name == 'CIFAR10':
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        elif name == 'MNIST':
            transform_list.append(transforms.Normalize(transforms.Normalize((0.1307,), (0.3081,))))
        elif name == 'CIFAR100':
            transform_list.append(transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)))
        elif name == 'FMNIST':
            transform_list.append(transforms.Normalize((0.2859,), (0.3530,)))
    transform = transforms.Compose(transform_list)
    if name == 'CIFAR10':
        trainset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)
        num_classes = 10
        rescale_fac = 2.7537
    elif name == 'MNIST':
        trainset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
        num_classes = 10
        rescale_fac = None
    elif name == 'CIFAR100':
        trainset = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transform)
        num_classes = 100
        rescale_fac = None
    elif name == 'FMNIST':
        trainset = datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=transform)
        num_classes = 10
        rescale_fac = None
    return trainset, testset, num_classes, rescale_fac

class IIDPartitioner():
    def __init__(self, num_clients, batch_size):
        self.num_clients = num_clients
        self.batch_size = batch_size

    def split_dataset(self, dataset: Dataset, shuff_gen = None) -> List[Subset]:
        indices = torch.randperm(len(dataset), generator=shuff_gen).tolist()
        n_batch = len(dataset) // self.batch_size
        split_size = n_batch // self.num_clients
        remainder = n_batch % self.num_clients

        subsets = []
        start_idx = 0
        for i in range(self.num_clients):
            end_idx = start_idx + (split_size + (1 if i < remainder else 0)) * self.batch_size 
            subsets.append(Subset(dataset, indices[start_idx:end_idx]))
            start_idx = end_idx

        return subsets
    
if __name__ == "__main__":
    # Example usage
    trainset, testset, num_classes = get_dataset('CIFAR10')