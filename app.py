import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
import gradio as gr
from mnist_denoising import train_model, sample_model, sample_new_image

app = Flask(__name__)
CORS(app)

def gradio_interface():
    def gradio_train_model(dataset_name, simulation, reconstruction_model, noise_level, epochs):
        model_type = "DeepAutoencoder"  # Update to use the DeepAutoencoder model
        model = train_model(dataset_name, noise_level, model_type, epochs=epochs)
        true_image, noisy_image, reconstructed_image = sample_model(dataset_name, noise_level, model_type)
        
        mse = compute_mse(true_image, reconstructed_image)
        
        return true_image, noisy_image, reconstructed_image, mse

    def gradio_sample_image(dataset_name, noise_level):
        true_image, noisy_image, reconstructed_image = sample_new_image(dataset_name, noise_level, "DeepAutoencoder")
        
        mse = compute_mse(true_image, reconstructed_image)
        
        return true_image, noisy_image, reconstructed_image, mse

    def compute_mse(true_image, reconstructed_image):
        true_image_tensor = transforms.ToTensor()(true_image)
        reconstructed_image_tensor = transforms.ToTensor()(reconstructed_image)
        mse = torch.mean((true_image_tensor - reconstructed_image_tensor) ** 2).item()
        return mse

    with gr.Blocks(css=".gradio-container {max-width: 1000px; margin: auto;} .gradio-container img {max-width: 300px;}") as demo:
        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                choices=["MNIST", "CIFAR10", "PathMNIST", "DermaMNIST", "OrganAMNIST"], label="Dataset"
            )
            simulation_dropdown = gr.Dropdown(choices=["AWGN"], label="Simulation Model")
            reconstruction_model_dropdown = gr.Dropdown(choices=["DeepAutoencoder"], label="Reconstruction Model")
            noise_level_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Noise Level")
            epochs_input = gr.Number(label="Epochs", value=50, precision=0)  # Default to 50 epochs
        
        with gr.Row():
            true_image_output = gr.Image(label="True Image", elem_id="true_image")
            noisy_image_output = gr.Image(label="Noisy Image", elem_id="noisy_image")
            reconstructed_image_output = gr.Image(label="Reconstructed Image", elem_id="reconstructed_image")
            mse_output = gr.Textbox(label="MSE", elem_id="mse_output", interactive=False)
        
        with gr.Row():
            train_button = gr.Button("Train Model")
            change_image_button = gr.Button("Change Image")

        train_button.click(
            gradio_train_model,
            inputs=[dataset_dropdown, simulation_dropdown, reconstruction_model_dropdown, noise_level_slider, epochs_input],
            outputs=[true_image_output, noisy_image_output, reconstructed_image_output, mse_output]
        )

        change_image_button.click(
            gradio_sample_image,
            inputs=[dataset_dropdown, noise_level_slider],
            outputs=[true_image_output, noisy_image_output, reconstructed_image_output, mse_output]
        )

    demo.launch()

if __name__ == '__main__':
    gradio_interface()
    app.run(debug=True)
