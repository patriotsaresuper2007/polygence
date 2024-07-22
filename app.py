from flask import Flask, request, jsonify
from flask_cors import CORS
import gradio as gr
from mnist_denoising import train_model

app = Flask(__name__)
CORS(app)

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        dataset_name = data['dataset_name']
        noise_level = data['noise_level']
        model_type = "Feed Forward Neural Network"  

        true_image, noisy_image, reconstructed_image = train_model(dataset_name, noise_level, model_type)
        return jsonify({
            "true_image": true_image,
            "noisy_image": noisy_image,
            "reconstructed_image": reconstructed_image
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def gradio_interface():
    def gradio_train_model(dataset_name, simulation, reconstruction_model, noise_level):
        true_image, noisy_image, reconstructed_image = train_model(dataset_name, noise_level, reconstruction_model)
        return true_image, noisy_image, reconstructed_image

    iface = gr.Interface(
        fn=gradio_train_model,
        inputs=[
            gr.Dropdown(choices=["MNIST", "CIFAR10", "PathMNIST", "DermaMNIST", "OrganAMNIST"], label="Dataset"),
            gr.Dropdown(choices=["AWGN"], label="Simulation Model"),
            gr.Dropdown(choices=["Feed Forward Neural Network"], label="Reconstruction Model"),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Noise Level")
        ],
        outputs=[gr.Image(label="True Image"), gr.Image(label="Noisy Image"), gr.Image(label="Reconstructed Image")]
    )

    iface.launch()

if __name__ == '__main__':
    gradio_interface()
    app.run(debug=True)
