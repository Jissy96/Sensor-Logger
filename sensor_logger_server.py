from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import os
import io
import tensorflow_hub as hub  # Import YAMNet model
import pyttsx3  # Import TTS for announcements

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

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to announce detected events via Bluetooth speaker
def speak_alert(message):
    """Uses TTS engine to announce detected events."""
    engine.say(message)
    engine.runAndWait()

# Function to process raw audio data using YAMNet
def process_audio_data(waveform, sr):
    """Extracts embeddings from audio using YAMNet and makes a prediction."""
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

    # Announce the detected sound
    alert_message = f"{predicted_label} detected!"
    print(alert_message)  # Log the event
    speak_alert(alert_message)  # Announce via Bluetooth speaker

    return predicted_label

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file to the uploads directory
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Load the audio file using librosa
        audio_data, sr = librosa.load(file_path, sr=None)

        # Process and predict the audio
        prediction = process_audio_data(audio_data, sr)

        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)