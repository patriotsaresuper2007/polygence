import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from medmnist.dataset import PathMNIST, DermaMNIST, OrganAMNIST

def get_dataset(dataset_name):
    transform = transforms.Compose([transforms.ToTensor()])
    root_dir = 'data'
    
    if dataset_name == "MNIST":
        return datasets.MNIST(root=os.path.join(root_dir, 'mnist'), train=True, transform=transform, download=True)
    elif dataset_name == "CIFAR10":
        return datasets.CIFAR10(root=os.path.join(root_dir, 'cifar10'), train=True, transform=transform, download=True)
    elif dataset_name == "PathMNIST":
        dataset_dir = os.path.join(root_dir, 'pathmnist')
        os.makedirs(dataset_dir, exist_ok=True)
        return PathMNIST(root=dataset_dir, split='train', transform=transform, download=True)
    elif dataset_name == "DermaMNIST":
        dataset_dir = os.path.join(root_dir, 'dermamnist')
        os.makedirs(dataset_dir, exist_ok=True)
        return DermaMNIST(root=dataset_dir, split='train', transform=transform, download=True)
    elif dataset_name == "OrganAMNIST":
        dataset_dir = os.path.join(root_dir, 'organamnist')
        os.makedirs(dataset_dir, exist_ok=True)
        return OrganAMNIST(root=dataset_dir, split='train', transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset")

def add_awgn_noise(image, noise_level=0.1):
    noise = noise_level * torch.randn_like(image)
    noisy_image = image + noise
    return noisy_image
