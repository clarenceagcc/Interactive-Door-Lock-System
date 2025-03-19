import tkinter as tk
from tkinter import messagebox, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import io
import torch
import numpy as np
import threading
import sounddevice as sd
from config import setup_database, save_image_to_db, fetch_images, delete_image_from_db
from wake_word_model import detect_wake_word  # Import wake word detection function
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import numpy as np
import sounddevice as sd

# Initialize the database
setup_database()

# Initialize the main Tkinter window
window = tk.Tk()
window.title("Capture Image and Save to MariaDB")
window.geometry("1024x768")

# Global variables
cap = None
image_label = None
captured_image = None
webcam_resolution = (640, 480)
wake_word_detected = False

class WakeWordModel(nn.Module):
    def __init__(self):
        super(WakeWordModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)  # Adjusted for Adaptive Pooling Output
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wake_word_model = WakeWordModel().to(device)
wake_word_model.load_state_dict(torch.load("wake_word_model (1).pth", map_location=device))
wake_word_model.eval()

# **Audio Settings**
SAMPLE_RATE = 16000  # Should match training setup
DURATION = 1.0  # 1 second buffer
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)  # Buffer size in samples
THRESHOLD = 0.7  # Confidence threshold for wake word detection

# Mel Spectrogram Transform (Fixed Parameters)
transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=64, n_fft=400, hop_length=160
).to(device)

# **Audio Buffer (To store 1-second audio chunks)**
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# **Process Audio for Model**
def process_audio(audio_data):
    waveform = torch.tensor(audio_data, dtype=torch.float32).to(device)
    waveform = waveform.unsqueeze(0)  # [1, samples]
    
    mel_spec = transform(waveform)  # [1, 64, time]
    mel_spec = mel_spec.unsqueeze(0)  # [1, 1, 64, time]

    return mel_spec

# **Wake Word Detection**
def detect_wake_word(audio_data):
    mel_spec = process_audio(audio_data)
    with torch.no_grad():
        output = wake_word_model(mel_spec)
        prediction = output.item()
    return prediction

# **Audio Streaming Callback**
def callback(indata, frames, time, status):
    global audio_buffer

    if status:
        print(f"Status: {status}")  # Handle stream errors

    # Shift buffer left and add new data (Sliding window)
    audio_buffer[:-frames] = audio_buffer[frames:]
    audio_buffer[-frames:] = indata[:, 0]  # Use first channel if stereo

    # Detect wake word
    prediction = detect_wake_word(audio_buffer)

    if prediction > THRESHOLD:
        print("🔥 Wake Word Detected! 🔥")
        start_camera()  # Call Tkinter function to start the camera

# **Start Audio Stream**
def start_audio_stream():
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
        print("Listening for wake word...")
        while True:
            pass  # Keep the stream alive

# Run the wake word detection
if __name__ == "__main__":
    start_audio_stream()

# Start the camera
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])
        capture_frame()

# Continuously capture frames from the camera
def capture_frame():
    global cap, image_label
    if cap:
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            if image_label is None:
                image_label = Label(image_frame, image=imgtk)
                image_label.pack()
            else:
                image_label.config(image=imgtk)
                image_label.image = imgtk
        window.after(10, capture_frame)

# Capture and store the current frame
def take_snapshot():
    global cap, captured_image, image_label
    if cap:
        ret, frame = cap.read()
        if ret:
            captured_image = cv2.imencode('.png', frame)[1].tobytes()
            img = Image.open(io.BytesIO(captured_image))
            imgtk = ImageTk.PhotoImage(img)
            if image_label is not None:
                image_label.config(image=imgtk)
                image_label.image = imgtk
            messagebox.showinfo("Image Captured", "Image captured successfully!")

# Save the captured image to the database
def save_to_database():
    global captured_image
    if captured_image is not None:
        save_image_to_db(captured_image)
        messagebox.showinfo("Success", "Image saved successfully!")
    else:
        messagebox.showwarning("No Image", "Please capture an image first.")

# Delete an image from the database
def delete_image(image_id):
    delete_image_from_db(image_id)
    messagebox.showinfo("Deleted", f"Image {image_id} deleted successfully!")
    display_saved_images()  # Refresh the image display

# Display saved images from the database
def display_saved_images():
    global cap, image_label
    if cap:
        cap.release()
        cap = None
    if image_label:
        image_label.pack_forget()
    for widget in image_frame.pack_slaves():
        widget.destroy()
    
    images = fetch_images()
    for img_id, name, img_data in images:
        img = Image.open(io.BytesIO(img_data))
        img = img.resize(webcam_resolution)
        imgtk = ImageTk.PhotoImage(img)
        
        # Create a frame for each image to hold the image and the delete button
        img_frame = Frame(image_frame)
        img_frame.pack(pady=5)
        
        img_label = Label(img_frame, image=imgtk)
        img_label.image = imgtk
        img_label.pack(side=tk.LEFT)
        
        delete_button = Button(img_frame, text="Delete", command=lambda img_id=img_id: delete_image(img_id))
        delete_button.pack(side=tk.LEFT, padx=5)

# Function to continuously listen for wake word in a background thread
def listen_for_wake_word():
    global wake_word_detected
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data = indata.flatten()
        prediction = detect_wake_word(audio_data)
        
        if prediction > 0.5:  # Threshold for wake word detection
            wake_word_detected = True
            print("Wake word detected!")
            start_camera()

    # Open an audio stream
    with sd.InputStream(callback=audio_callback, samplerate=16000, channels=1):
        while True:
            if wake_word_detected:
                break  # Stop listening after wake word is detected

# Start wake word detection in a separate thread
wake_word_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
wake_word_thread.start()

# Layout with frames
control_frame = Frame(window)
control_frame.pack(side=tk.LEFT, fill=tk.Y)
image_frame = Frame(window)
image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

# Control Buttons
Button(control_frame, text="Start Camera", command=start_camera).pack(pady=5)
Button(control_frame, text="Capture Image", command=take_snapshot).pack(pady=5)
Button(control_frame, text="Save to Database", command=save_to_database).pack(pady=5)
Button(control_frame, text="Display Saved Images", command=display_saved_images).pack(pady=5)
Button(control_frame, text="Face Recognition", command=display_saved_images).pack(pady=5)

# Run the Tkinter GUI
window.mainloop()
