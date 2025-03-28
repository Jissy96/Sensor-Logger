import tensorflow as tf
import numpy as np
import librosa
import tensorflow_hub as hub

# Load YAMNet model
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')


# Function to classify sound
def classify_sound(file_path):
    """Loads an audio file, extracts features, and predicts its category."""
    waveform, sr = librosa.load(file_path, sr=16000)

    # 🔹 Ensure waveform is 1D
    waveform = np.array(waveform, dtype=np.float32).flatten()

    # 🔹 Make prediction
    scores, embeddings, spectrogram = yamnet(waveform)

    # 🔹 Get top prediction
    predicted_index = tf.argmax(scores, axis=-1).numpy()
    confidence = tf.reduce_max(scores, axis=-1).numpy()

    return predicted_index, confidence


# Test on an audio file
sample_file = "dataset/fire_alarm/554776__oddbro2020__old-broken-fire-alarm.wav"  # Update with actual file path
if sample_file:
    predicted_class, confidence = classify_sound(sample_file)
    print(f"Predicted Class Index: {predicted_class}, Confidence: {confidence[0]:.2f}")  # ✅ Fixed!
else:
    print("Test file not found. Please check the path.")
