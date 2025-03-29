from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import os
import io
import tensorflow_hub as hub  # Import YAMNet model

# Load the trained model
model = tf.keras.models.load_model("fine_tuned_yamnet.h5")

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Categories for the prediction
CATEGORIES = ["fire_alarm", "glass_break", "baby_crying", "doorbell", "gunshot"]

# Create Flask app
app = Flask(__name__)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to process raw audio data using YAMNet

def process_audio_data(waveform, sr):
    """Extracts embeddings from audio using YAMNet."""
    waveform = waveform.astype(np.float32)  # Ensure float32 type
    waveform = waveform / np.max(np.abs(waveform))  # Normalize audio

    # Get YAMNet embeddings
    scores, embeddings, spectrogram = yamnet_model(waveform)
    embeddings = embeddings.numpy()  # Convert Tensor to NumPy array

    # Take the mean across time steps to match the model input shape
    embeddings = np.mean(embeddings, axis=0).reshape(1, -1)

    # Make a prediction
    predictions = model.predict(embeddings)
    predicted_label = CATEGORIES[np.argmax(predictions)]

    return predicted_label

# Route to handle live streaming from the phone
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the raw audio bytes from the request
        audio_data = request.data

        # Convert byte data to numpy array
        waveform, sr = librosa.load(io.BytesIO(audio_data), sr=None)

        # Process and predict
        prediction = process_audio_data(waveform, sr)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Load the audio file using librosa
    audio_data, sr = librosa.load(file_path, sr=None)

    # Process the audio file
    prediction = process_audio_data(audio_data, sr)

    return jsonify({'prediction': prediction}), 200

# Run Flask server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)