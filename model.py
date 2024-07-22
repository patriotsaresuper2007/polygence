import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from medmnist.dataset import PathMNIST, DermaMNIST, OrganAMNIST

class DenseNet(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_channels_list, activation='relu'):
        super(DenseNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_size = int(torch.prod(torch.tensor(input_shape)))
        self.output_size = int(torch.prod(torch.tensor(output_shape)))

        layers = []
        if isinstance(activation, str):
            activation = [activation] * len(hidden_channels_list)
        elif isinstance(activation, list) and len(activation) != len(hidden_channels_list):
            raise ValueError("Length of activation functions list must match the length of hidden_channels_list")

        in_size = self.input_size
        for out_size, act in zip(hidden_channels_list, activation):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(self.get_activation(act))
            in_size = out_size

        layers.append(nn.Linear(in_size, self.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch_shape = x.shape[:-len(self.input_shape)]
        batch_size = int(torch.prod(torch.tensor(batch_shape)))
        x = x.view(batch_size, self.input_size)  

        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d) and batch_size == 1:
                continue  
            x = layer(x)

        x = x.view(*batch_shape, *self.output_shape)
        return x

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'prelu':
            return nn.PReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

def get_densenet_model(model_type, input_shape):
    if model_type == "DenseNetSmall":
        return DenseNet(input_shape=input_shape, output_shape=input_shape, hidden_channels_list=[1024])
    elif model_type == "DenseNetLarge":
        return DenseNet(input_shape=input_shape, output_shape=input_shape, hidden_channels_list=[1024, 2048, 1024])
    elif model_type == "Feed Forward Neural Network":
        return DenseNet(input_shape=input_shape, output_shape=input_shape, hidden_channels_list=[1024, 2048, 1024])
    else:
        raise ValueError("Unsupported model type")

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