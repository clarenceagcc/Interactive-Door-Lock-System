import torch
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import numpy as np
import sounddevice as sd
import tkinter as tk
from threading import Thread
import cv2
from PIL import Image, ImageTk
import os

# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("Wake Word Detection with Camera Feed")

# Camera settings
webcam_resolution = (640, 480)
cap = None

# Create directory for saved faces
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

# Label to display the camera feed
camera_label = tk.Label(root)
camera_label.pack()

# --- Start the camera and display feed in the main window ---
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])
        update_camera_feed()

def update_camera_feed():
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)
        camera_label.after(10, update_camera_feed)

# --- Face Registration ---
def register_face():
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_filename = f"saved_faces/face_{len(os.listdir('saved_faces')) + 1}.png"
            cv2.imwrite(face_filename, frame)
            print(f"Face saved as {face_filename}")

# --- Display Saved Faces ---
def display_saved_faces():
    face_window = tk.Toplevel(root)
    face_window.title("Saved Faces")
    for i, filename in enumerate(os.listdir("saved_faces")):
        face_path = os.path.join("saved_faces", filename)
        img = Image.open(face_path)
        img = img.resize((100, 100))
        imgtk = ImageTk.PhotoImage(img)
        label = tk.Label(face_window, image=imgtk)
        label.image = imgtk
        label.grid(row=i // 5, column=i % 5, padx=5, pady=5)

# --- Wake Word Model ---
class WakeWordModel(nn.Module):
    def __init__(self):
        super(WakeWordModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wake_word_model = WakeWordModel().to(device)
wake_word_model.load_state_dict(torch.load("wake_word_model.pth", map_location=device))
wake_word_model.eval()

# --- Audio Configuration ---
SAMPLE_RATE = 16000
DURATION = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
THRESHOLD = 0.99
transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=400, hop_length=160).to(device)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# --- Audio Processing ---
def process_audio(audio_data):
    waveform = torch.tensor(audio_data, dtype=torch.float32).to(device)
    mel_spec = transform(waveform.unsqueeze(0)).unsqueeze(0)
    return mel_spec

def detect_wake_word(audio_data):
    mel_spec = process_audio(audio_data)
    with torch.no_grad():
        return wake_word_model(mel_spec).item()

# --- Audio Callback ---
def callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer[:-frames] = audio_buffer[frames:]
    audio_buffer[-frames:] = indata[:, 0]
    
    prediction = detect_wake_word(audio_buffer)
    print(f"Wake word probability: {prediction}")  # Debugging line

    if prediction > THRESHOLD:
        print("Wake word detected! Starting camera...")  # Debugging line
        root.after(0, start_camera)


# --- Start Audio Stream ---
def start_audio_stream():
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
        print("Listening for wake word...")
        sd.sleep(1000000)

# --- Add Buttons ---
button_frame = tk.Frame(root)
button_frame.pack()

register_button = tk.Button(button_frame, text="Register Face", command=register_face)
register_button.pack(side=tk.LEFT, padx=10, pady=10)

display_button = tk.Button(button_frame, text="Display Saved Faces", command=display_saved_faces)
display_button.pack(side=tk.LEFT, padx=10, pady=10)

# --- Start Application ---
if __name__ == "__main__":
    root.geometry("640x580")
    label = tk.Label(root, text="Waiting for wake word...")
    label.pack(pady=10)

    audio_thread = Thread(target=start_audio_stream, daemon=True)
    audio_thread.start()

    root.mainloop()

    if cap is not None and cap.isOpened():
        cap.release()
