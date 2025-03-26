import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, Subset
from typing import List

def dataset(name, download=True):
    data_dir = f"/tmp/data"
    if name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)
        return trainset, testset, 10
    elif name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
        return trainset, testset, 10
    elif name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        trainset = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transform)
        return trainset, testset, 100
    elif name == 'FMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2859,), (0.3530,)),
        ])
        trainset = datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform)
        testset = datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=transform)
        return trainset, testset, 10

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
    
