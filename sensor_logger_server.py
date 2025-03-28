from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa

# Load the trained model
model = tf.keras.models.load_model("fine_tuned_yamnet.h5")

# Categories for the prediction
CATEGORIES = ["fire_alarm", "glass_break", "baby_crying", "doorbell", "gunshot"]

# Create Flask app
app = Flask(__name__)

# Function to process the received audio data
def process_audio_data(waveform):
    features = np.expand_dims(waveform.mean(axis=0), axis=0)
    predictions = model.predict(features)
    predicted_label = CATEGORIES[np.argmax(predictions)]
    return predicted_label

# Route for receiving audio data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get audio data from the request (assume the data is sent in raw byte format)
        audio_data = request.get_data()  # The audio data in byte format

        # Convert byte data to numpy array (this will depend on how the app sends it)
        waveform = np.frombuffer(audio_data, dtype=np.float32)

        # Process and predict
        prediction = process_audio_data(waveform)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
