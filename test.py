import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasetsx
import matplotlib.pyplot as plt
import pdb
from medmnist import INFO, Evaluator
from medmnist.dataset import PathMNIST, DermaMNIST, OrganAMNIST

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the DenseNet model
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

def train_model(dataset_name, noise_level, model_type, epochs=50, batch_size=16, learning_rate=1e-4):
    dataset = get_dataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if dataset_name == "MNIST":
        input_shape = (1, 28, 28)
    elif dataset_name == "CIFAR10":
        input_shape = (3, 32, 32)
    elif dataset_name in ["PathMNIST", "DermaMNIST", "OrganAMNIST"]:
        input_shape = (3, 28, 28)

    model = get_densenet_model(model_type, input_shape).to(device)  

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  

    for epoch in range(epochs):
        model.train()
        for images, _ in dataloader:
            images = images.to(device)
            noisy_images = add_awgn_noise(images, noise_level).to(device)

            outputs = model(noisy_images)
            loss = loss_fn(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    model_save_path = f"models/{dataset_name}_{model_type}_noise_{noise_level}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    return model

# Testing
if __name__ == "__main__":
    dataset_name = "DermaMNIST"
    noise_level = 0.05
    model_type = "Feed Forward Neural Network"

    model = train_model(dataset_name, noise_level, model_type, epochs=10)

    dataset = get_dataset(dataset_name)
    sample_image, _ = dataset[0]

    sample_image = sample_image.unsqueeze(0).to(device)
    noisy_sample_image = add_awgn_noise(sample_image, noise_level)

    pdb.set_trace()

    to_pil = transforms.ToPILImage()
    sample_image_pil = to_pil(sample_image.cpu().squeeze(0))
    noisy_image_pil = to_pil(noisy_sample_image.cpu().squeeze(0))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(sample_image_pil, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(noisy_image_pil, cmap='gray')
    axs[1].set_title('Noisy Image')
    axs[1].axis('off')

    plt.show()


    model.eval()
    with torch.no_grad():
        reconstructed_image = model(noisy_sample_image).squeeze(0)

    reconstructed_image_pil = to_pil(reconstructed_image.cpu())


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(sample_image_pil, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(noisy_image_pil, cmap='gray')
    axs[1].set_title('Noisy Image')
    axs[1].axis('off')

    axs[2].imshow(reconstructed_image_pil, cmap='gray')
    axs[2].set_title('Reconstructed Image')
    axs[2].axis('off')

    plt.show()
