import base64
from io import BytesIO
from PIL import Image
import uuid
from datetime import datetime

def image_to_base64(image_bytes):
    """Converts image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")

def get_file_extension(filename):
    """Gets the file extension from a filename."""
    if '.' in filename:
        return filename.split('.')[-1].lower()
    return "" # Return empty string if no extension

def load_image_from_bytes(image_bytes):
    """Loads an image from bytes using Pillow."""
    return Image.open(BytesIO(image_bytes))

def generate_uuid():
    """Generates a unique identifier."""
    return str(uuid.uuid4())

def get_current_timestamp():
    """Gets the current timestamp in a readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")