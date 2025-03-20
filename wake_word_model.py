import torch
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import numpy as np
import sounddevice as sd
import tkinter as tk
from threading import Thread, Event
import cv2
from PIL import Image, ImageTk
import os

# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("Wake Word Detection with Camera Feed")
audio_paused = Event()  # Event to control audio thread pause

# Camera settings
webcam_resolution = (640, 480)
cap = None

# Create directory for saved faces
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

# Label to display the camera feed
camera_label = tk.Label(root)
camera_label.pack()

# --- Start the camera and display feed ---
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change to 1 if using an external camera
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
        root.after(10, update_camera_feed)  # Ensure Tkinter updates the UI

# --- Face Registration ---
def register_face():
    global cap, face_window
    audio_paused.set()  # Pause audio thread

    # Release the main camera feed to avoid conflicts
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None  

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])

    face_window = tk.Toplevel(root)
    face_window.title("Face Registration")

    face_label = tk.Label(face_window)
    face_label.pack()

    save_button = tk.Button(face_window, text="Save Face", command=save_face)
    save_button.pack(pady=10)

    close_button = tk.Button(face_window, text="Close", command=close_face_registration)
    close_button.pack(pady=10)

    update_face_feed(face_label)

def update_face_feed(face_label):
    if face_window.winfo_exists():  # Ensure window still exists
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                face_label.imgtk = imgtk
                face_label.configure(image=imgtk)
        face_window.after(10, lambda: update_face_feed(face_label))

def save_face():
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_filename = f"saved_faces/face_{len(os.listdir('saved_faces')) + 1}.png"
            cv2.imwrite(face_filename, frame)
            print(f"Face saved as {face_filename}")

def close_face_registration():
    global cap, face_window
    if face_window.winfo_exists():  # Ensure window exists before closing
        face_window.destroy()

    # Don't release the camera, just return to the main feed
    audio_paused.clear()  # Resume audio detection

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
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout to reduce overfitting
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wake_word_model = WakeWordModel().to(device)
wake_word_model.load_state_dict(torch.load("476998.pth", map_location=device))
wake_word_model.eval()

# --- Audio Configuration ---
SAMPLE_RATE = 16000
DURATION = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
THRESHOLD = 0.78

# Transform for mel spectrogram
transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=400, hop_length=160).to(device)

# Initialize an empty buffer
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# --- Audio Processing ---
def process_audio(audio_data):
    waveform = torch.tensor(audio_data, dtype=torch.float32).to(device)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Ensure batch dimension
    mel_spec = transform(waveform).unsqueeze(0)  # Add channel dimension for Conv2D
    return mel_spec

def detect_wake_word(audio_data):
    mel_spec = process_audio(audio_data)
    with torch.no_grad():
        output = wake_word_model(mel_spec)
        return torch.sigmoid(output).item()  # Sigmoid for probability

# --- Audio Callback ---
def callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"Audio callback error: {status}")
    
    if audio_paused.is_set():
        return  # Pause audio detection when registering a face

    audio_buffer[:-frames] = audio_buffer[frames:]
    audio_buffer[-frames:] = indata[:, 0]
    
    prediction = detect_wake_word(audio_buffer)
    print(f"Wake word probability: {prediction}")

    if prediction > THRESHOLD:
        print("Wake word detected! Starting camera...")
        root.after(0, start_camera)

# --- Start Audio Stream ---
def start_audio_stream():
    stream = sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
    stream.start()  # Start non-blocking stream
    print("Listening for wake word...")

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

    Thread(target=start_audio_stream, daemon=True).start()  # Run audio in separate thread

    root.mainloop()

    if cap is not None and cap.isOpened():
        cap.release()
