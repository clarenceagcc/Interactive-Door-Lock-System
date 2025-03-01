import streamlit as st
from PIL import Image
import io
import os
from config import fetch_images, delete_image_from_db  # Assuming you have a delete function

# Function to fetch and display saved images (Placeholder for actual database logic)
def load_saved_faces():
    images = fetch_images()
    for img_id, name, img_data in images:
        # Display image with a delete button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader(name)
            st.image(io.BytesIO(img_data), caption=name, use_container_width=True)
        with col2:
            if st.button("Delete", key=img_id):
                delete_image_from_db(img_id)  # Assuming this function deletes the image from the database
                st.success(f"Image '{name}' deleted successfully.")
                st.rerun()  # Refresh the page after deletion

    return images
# Display title and instructions
st.title("View Saved Faces")

# Load and display saved face images
saved_faces = load_saved_faces()
