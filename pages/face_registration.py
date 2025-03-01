import streamlit as st
import io
from config import setup_database, save_image_to_db, fetch_images

# Run the database setup at startup
setup_database()

# Streamlit UI
st.title("Capture Image from Webcam and Save to MariaDB")

# Capture image from webcam
captured_image = st.camera_input("Capture Image")

if captured_image:
    if st.button("Save to Database"):
        save_image_to_db(captured_image)
        st.success("Image saved successfully!")

st.header("Saved Images")
images = fetch_images()
for img_id, name, img_data in images:
    st.subheader(name)
    st.image(io.BytesIO(img_data), caption=name, use_column_width=True)
