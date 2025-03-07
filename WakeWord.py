import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import numpy as np
import sounddevice as sd

# Load trained model
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
model = WakeWordModel().to(device)
model.load_state_dict(torch.load("wake_word_model.pth", map_location=device))
model.eval()

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
        output = model(mel_spec)  # Model expects [batch, channel, height, width]
        prediction = output.item()
    
    return prediction

# Real-Time Audio Streaming
def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")  # Print status errors if any
    
    audio_data = indata[:, 0]  # Use first channel if stereo
    prediction = detect_wake_word(audio_data)
    
    print("\n==== Incoming Audio ====")
    print(f"Captured Audio Data (First 5 samples): {np.round(audio_data[:5], 6)}")  # Round values for better readability
    print(f"Model Prediction: {prediction:.6f}")  # Print prediction score with 6 decimals
    print("========================\n")
    
    if prediction > 0.7:  # Threshold for detection
        print("🔥 Wake Word Detected! 🔥\n")

# Start Recording
print("Listening for wake word...")
with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
    input("Press ENTER to stop.\n")
