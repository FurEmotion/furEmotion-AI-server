from flask import Flask, request, jsonify
import torch
import numpy as np
import librosa
import io
import soundfile as sf
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import os

app = Flask(__name__)
state_list = ['hostile', 'play', 'relax', 'whining']

# Set up model directory
model_dir = os.path.join(os.getcwd(), 'ast_model_pytorch_dog')
print(f"Model directory: {model_dir}")

# Check if the model directory exists
if not os.path.isdir(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

# Load the model and feature extractor
try:
    model = ASTForAudioClassification.from_pretrained(
        model_dir,
        num_labels=len(state_list),
        ignore_mismatched_sizes=True
    )
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_dir)
    print("Model and feature extractor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load the state (label) names
# Assuming you have a state_list.txt file in the model directory
label_file = os.path.join(model_dir, 'state_list.txt')
if os.path.exists(label_file):
    with open(label_file, 'r') as f:
        state_list = [line.strip() for line in f]
    print(f"Loaded state list: {state_list}")
else:
    # Define state_list manually if the file doesn't exist
    # Replace with your actual state names
    print(f"Using default state list: {state_list}")

# Move model to device and set to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


@app.route('/', methods=['POST'])
def classify_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the audio file
        audio_bytes = file.read()
        # Load audio using soundfile
        audio_data, samplerate = sf.read(
            io.BytesIO(audio_bytes), dtype='float32')
        print(
            f"Original sample rate: {samplerate}, Audio shape: {audio_data.shape}")

        # Resample to 16000 Hz if necessary
        target_samplerate = 16000
        if samplerate != target_samplerate:
            audio_data = librosa.resample(
                audio_data, orig_sr=samplerate, target_sr=target_samplerate)
            samplerate = target_samplerate
            print(f"Resampled to {target_samplerate} Hz")

        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print("Converted audio to mono")

        # Process audio
        inputs = feature_extractor(
            audio_data, sampling_rate=samplerate, return_tensors="pt", padding=True
        )
        input_values = inputs['input_values'].to(device)

        # Get logits and apply softmax
        with torch.no_grad():
            outputs = model(input_values=input_values)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]

        # Map probabilities to state names
        result = {state: float(prob)
                  for state, prob in zip(state_list, probabilities)}
        print(f"Prediction result: {result}")

        return jsonify(result)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)
