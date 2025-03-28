import numpy as np
import tensorflow as tf
import librosa
import tensorflow_hub as hub

# Load the trained model
model = tf.keras.models.load_model("fine_tuned_yamnet.h5")

# Categories
CATEGORIES = ["fire_alarm", "glass_break", "baby_crying", "doorbell", "gunshot"]

# Load YAMNet model
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet = hub.load(yamnet_model_handle)


# Function to extract features using YAMNet
def extract_features(file_path):
    """Extracts audio embeddings using YAMNet."""
    waveform, sr = librosa.load(file_path, sr=16000)

    # Get the embeddings from YAMNet
    scores, embeddings, spectrogram = yamnet(waveform)

    # Return the mean of the embeddings (1024-dimensional feature vector)
    return embeddings.numpy().mean(axis=0)


# Function to predict a sound file
def predict_sound(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    predictions = model.predict(features)
    predicted_label = CATEGORIES[np.argmax(predictions)]

    print(f"Predicted Sound: {predicted_label}")


# Test on a sample file
predict_sound("dataset/fire_alarm/554776__oddbro2020__old-broken-fire-alarm.wav")
