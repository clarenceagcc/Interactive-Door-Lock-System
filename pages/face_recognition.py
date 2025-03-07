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
st.title("Face Recognition")

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

