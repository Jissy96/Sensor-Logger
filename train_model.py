import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load YAMNet model
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet = hub.load(yamnet_model_handle)

# Categories (Make sure they match your dataset folders)
CATEGORIES = ["fire_alarm", "glass_break", "baby_crying", "doorbell", "gunshot"]


# Function to load and extract embeddings from YAMNet
def extract_features(file_path):
    """Extracts audio embeddings using YAMNet."""
    waveform, sr = librosa.load(file_path, sr=16000)

    # No need to add an extra batch dimension here
    scores, embeddings, spectrogram = yamnet(waveform)

    return embeddings.numpy().mean(axis=0)  # Use the mean embedding


# Load dataset and labels
X = []
y = []

for label, category in enumerate(CATEGORIES):
    folder = f"dataset/{category}"
    for file_name in os.listdir(folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder, file_name)
            features = extract_features(file_path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple Neural Network
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),  # Input layer
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(CATEGORIES), activation="softmax")  # Output layer (for classification)
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
print("Training model...")
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save("fine_tuned_yamnet.h5")
print("Model training complete. Saved as fine_tuned_yamnet.h5")

