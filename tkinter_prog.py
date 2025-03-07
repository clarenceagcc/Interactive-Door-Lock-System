import tkinter as tk
from tkinter import messagebox, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import io
from config import setup_database, save_image_to_db, fetch_images, delete_image_from_db

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

# Start the camera
def start_camera():
    global cap
    # Clear any previously displayed images before starting the camera feed
    clear_image_frame()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])
    capture_frame()

# Clear the image_frame (where saved images are displayed)
def clear_image_frame():
    for widget in image_frame.pack_slaves():
        widget.destroy()

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

window.mainloop()
