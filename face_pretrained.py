import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from scipy.spatial.distance import cosine

# Initialize the InsightFace app
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])  # Using Buffalo I (lightweight)
app.prepare(ctx_id=0, det_size=(640, 640))  # Set detection size

print("Buffalo I model loaded successfully!")

import matplotlib.pyplot as plt
def extract_embedding(img_path):
    """Extracts a 512-D face embedding using Buffalo I"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    faces = app.get(img)

    if len(faces) == 0:
        print(f"No face detected in {img_path}")
        return None

    # Get the first detected face embedding
    return faces[0].embedding

def verify_faces(img1_path, img2_path, threshold=0.5):
    """Compares two face embeddings and verifies if they belong to the same person"""
    emb1 = extract_embedding(img1_path)
    emb2 = extract_embedding(img2_path)

    if emb1 is None or emb2 is None:
        print("Face not detected in one or both images!")
        return

    # Compute cosine similarity (lower = more similar)
    distance = cosine(emb1, emb2)

    print(f"Cosine Distance: {distance:.4f}")

    if distance < threshold:
        print("✅ Faces Match (Same Person)")
    else:
        print("❌ Faces Do NOT Match (Different People)")

def plot_images(img1_path, img2_path):
    # Read and load images for displaying
    img1 = plt.imread(img1_path)
    img2 = plt.imread(img2_path)

    # Plot the two images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title("Image 1")

    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title("Image 2")

    plt.show()

img1_path = "C:/Users/agccc/OneDrive/Pictures/Camera Roll 1/WIN_20250321_14_46_31_Pro.jpg"
img2_path = "C:/Users/agccc/OneDrive/Pictures/Camera Roll 1/WIN_20250321_14_46_36_Pro.jpg"

verify_faces(img1_path, img2_path)