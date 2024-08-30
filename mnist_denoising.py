import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import add_awgn_noise, get_dataset
import torchvision.transforms as transforms
from autoencoder import DeepAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(dataset_name, noise_level, model_type, epochs=50, batch_size=16, learning_rate=1e-4):
    dataset = get_dataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if dataset_name == "MNIST":
        input_shape = (1, 28, 28)
    elif dataset_name == "CIFAR10":
        input_shape = (3, 32, 32)
    elif dataset_name in ["PathMNIST", "DermaMNIST", "OrganAMNIST"]:
        input_shape = (3, 28, 28)

    model = DeepAutoencoder(input_shape).to(device)

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

def sample_model(dataset_name, noise_level, model_type):
    model_save_path = f"models/{dataset_name}_{model_type}_noise_{noise_level}.pth"

    dataset = get_dataset(dataset_name)
    if dataset_name == "MNIST":
        input_shape = (1, 28, 28)
    elif dataset_name == "CIFAR10":
        input_shape = (3, 32, 32)
    elif dataset_name in ["PathMNIST", "DermaMNIST", "OrganAMNIST"]:
        input_shape = (3, 28, 28)

    model = DeepAutoencoder(input_shape).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    random_index = random.randint(0, len(dataset) - 1)
    sample_image, _ = dataset[random_index]
    noisy_sample_image = add_awgn_noise(sample_image.unsqueeze(0).to(device), noise_level)
    reconstructed_image = model(noisy_sample_image).detach().squeeze(0)

    to_pil = transforms.ToPILImage()
    sample_image_pil = to_pil(sample_image)
    noisy_image_pil = to_pil(noisy_sample_image.cpu().squeeze(0))
    reconstructed_image_pil = to_pil(reconstructed_image.cpu())

    return sample_image_pil, noisy_image_pil, reconstructed_image_pil

def sample_new_image(dataset_name, noise_level, model_type):
    model_save_path = f"models/{dataset_name}_{model_type}_noise_{noise_level}.pth"

    if not os.path.exists(model_save_path):
        print(f"Model not found at {model_save_path}, training a new model...")
        train_model(dataset_name, noise_level, model_type)

    dataset = get_dataset(dataset_name)
    if dataset_name == "MNIST":
        input_shape = (1, 28, 28)
    elif dataset_name == "CIFAR10":
        input_shape = (3, 32, 32)
    elif dataset_name in ["PathMNIST", "DermaMNIST", "OrganAMNIST"]:
        input_shape = (3, 28, 28)

    model = DeepAutoencoder(input_shape).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    random_index = random.randint(0, len(dataset) - 1)
    sample_image, _ = dataset[random_index]
    noisy_sample_image = add_awgn_noise(sample_image.unsqueeze(0).to(device), noise_level)
    reconstructed_image = model(noisy_sample_image).detach().squeeze(0)

    to_pil = transforms.ToPILImage()
    sample_image_pil = to_pil(sample_image)
    noisy_image_pil = to_pil(noisy_sample_image.cpu().squeeze(0))
    reconstructed_image_pil = to_pil(reconstructed_image.cpu())

    return sample_image_pil, noisy_image_pil, reconstructed_image_pil
