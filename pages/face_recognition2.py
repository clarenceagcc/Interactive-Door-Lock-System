import streamlit as st
from PIL import Image
import io
import os
from config import fetch_images
import torch
from torchvision import transforms
from model import SiameseNetworkBCE
import numpy as np
import cv2
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import sounddevice as sd


class WakeWordModel(nn.Module):
    def __init__(self):
        super(WakeWordModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load model and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wake_word_model = WakeWordModel().to(device)
wake_word_model.load_state_dict(torch.load("wake_word_model.pth", map_location=device))
wake_word_model.eval()

# Define Audio Parameters
SAMPLE_RATE = 16000  # Ensure it matches training sample rate
DURATION = 1  # 1-second audio clips
BUFFER_SIZE = SAMPLE_RATE * DURATION

# Mel Spectrogram Transform (same as in training)
transform = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64).to(device)

# Audio Processing Function (Fixing Input Shape)
def process_audio(audio_data):
    waveform = torch.tensor(audio_data, dtype=torch.float32).to(device)
    waveform = waveform.unsqueeze(0)  # Add batch dimension -> [1, samples]

    mel_spec = transform(waveform)  # Convert to Mel Spectrogram -> [1, 64, Time]

    if len(mel_spec.shape) == 3:
        mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension -> [1, 1, 64, Time]

    return mel_spec

# Wake Word Detection Function (Uses Fixed Processing)
def detect_wake_word(audio_data):
    mel_spec = process_audio(audio_data)  # Ensure correct shape

    with torch.no_grad():
        output = wake_word_model(mel_spec)  # Model expects [batch, channel, height, width]
        prediction = output.item()

    return prediction

wake_word_detected = False

# Real-Time Audio Streaming
def callback(indata, frames, time, status):
    global wake_word_detected
    if status:
        print(f"Status: {status}")  # Print status errors if any

    audio_data = indata[:, 0]  # Use first channel if stereo
    prediction = detect_wake_word(audio_data)

    if prediction > 0.7:  # Threshold for detection
        print("🔥 Wake Word Detected! 🔥\n")
        wake_word_detected = True

# --- Face Recognition ---

# Function to fetch and display saved images
def load_saved_faces():
    images = fetch_images()
    if images:
        # Assuming the first saved image is the one you want
        img_id, name, img_data = images[0]
        img = Image.open(io.BytesIO(img_data))
        st.subheader(name)
        st.image(img, caption=name, use_container_width=True)
        return img, img_id
    return None, None

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Change based on your input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Load the saved model
def load_model():
    checkpoint = torch.load('siamese_model.pth', map_location=torch.device('cpu'))
    model = SiameseNetworkBCE()
    model.load_state_dict(checkpoint['model_state_dict'])  # Ensure loading on CPU
    model.eval()
    return model

# Compare two images
def compare_faces(model, image1, image2, threshold=0.5):
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Pass the images through the model
    with torch.no_grad():
        output = model(image1, image2)

    similarity_score = output.item()
    st.write(f"Similarity score: {similarity_score}")

    # Check if faces match based on a threshold
    if similarity_score < threshold:
        st.success("The faces match!")
    else:
        st.error("The faces do not match.")

# Streamlit UI
st.title("Voice and Face Recognition")

# Initialize session state
if 'wake_word_detected' not in st.session_state:
    st.session_state.wake_word_detected = False

if not st.session_state.wake_word_detected:
    # Start Recording
    print("Listening for wake word...")
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
        st.write("Say the wake word to activate face recognition.")
        input("Press ENTER to stop wake word detection.\n")

if st.session_state.wake_word_detected:
    st.write("Wake word detected! Proceeding to face recognition.")
    # Load and display the first saved face image
    image1, img_id = load_saved_faces()

    # Webcam capture
    camera = st.camera_input("Capture Image of Face")

    # If a webcam image is captured
    if camera:
        # Load the captured webcam image
        img_bytes = camera.getvalue()
        image2 = Image.open(io.BytesIO(img_bytes))

        # Show the webcam image
        st.image(image2, caption="Captured Face", use_container_width=True)

        # Load the model
        model = load_model()

        # Compare the saved image with the captured face
        if image1 is not None:
            compare_faces(model, image1, image2)
else:
    st.write("Wake word not detected. Face recognition not activated.")