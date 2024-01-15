import base64
import io
from PIL import Image


with open('call.txt', 'r') as f:
    # Read the string from the file
    file_string = f.read().strip()
# Now base64_image is a base64-encoded string representing the image
    
image_data = base64.b64decode(file_string)

image_file = io.BytesIO(image_data)
image = Image.open(image_file).convert('RGB')