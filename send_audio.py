# Print the server response
print(response.json())
import requests
import sounddevice as sd
import numpy as np
import io
import librosa

# Flask server URL
url = "http://172.2.10.3:5000/predict"

# Function to capture and send audio data
def send_audio():
    duration = 3  # Record for 3 seconds
    sample_rate = 44100

    print("Recording audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for recording to finish

    # Convert numpy array to bytes
    audio_bytes = io.BytesIO()
    np.save(audio_bytes, audio_data)

    print("Sending data to server...")
    response = requests.post(url, data=audio_bytes.getvalue())

    # Print server response
    print(response.json())

# Run the function
send_audio()