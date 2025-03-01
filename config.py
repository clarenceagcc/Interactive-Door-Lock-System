import mariadb

# Database connection function
def get_connection():
    return mariadb.connect(
        host="localhost",
        user="root",       # Change if needed
        password="admin",  # Change if needed
    )

# Function to create database and table if not exists
def setup_database():
    conn = get_connection()
    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.execute("CREATE DATABASE IF NOT EXISTS image_db")
    cursor.execute("USE image_db")

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            image LONGBLOB
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


def delete_image_from_db(img_id):
    # Logic to delete the image from your database, e.g.:
    # Assuming you're using MariaDB or similar, here's a basic query example
    conn = mariadb.connect(
        host="localhost",
        user="root",
        password="admin",
        database="image_db"
    )
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images WHERE id = %s", (img_id,))
    conn.commit()
    cursor.close()
    conn.close()

# Function to save an image to the database
def save_image_to_db(image_file):
    conn = mariadb.connect(
        host="localhost",
        user="root",
        password="admin",
        database="image_db"
    )
    cursor = conn.cursor()
    image_data = image_file.read()
    query = "INSERT INTO images (name, image) VALUES (%s, %s)"
    cursor.execute(query, (image_file.name, image_data))
    conn.commit()
    cursor.close()
    conn.close()

# Function to fetch images from the database
def fetch_images():
    conn = mariadb.connect(
        host="localhost",
        user="root",
        password="admin",
        database="image_db"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, image FROM images")
    images = cursor.fetchall()
    cursor.close()
    conn.close()
    return images
