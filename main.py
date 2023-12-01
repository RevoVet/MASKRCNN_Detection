from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import functional as F


app = FastAPI()


def load_model():
    # placeholder function
    model = torch.load("revovet_HH_vask_initial_model_epoch_9.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()


def preprocess_image(image_file):
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_file)).convert("RGB")
    # Convert to tensor
    image = F.to_tensor(image)
    return image


def predict(image, model):
    with torch.no_grad():
        prediction = model([image])
    return prediction




@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Read image file
    image_contents = await file.read()
    
    # Preprocess the image
    image = preprocess_image(image_contents)

    # Get predictions
    predictions = predict(image, model)

    # Process predictions (for demonstration, just returning the first prediction)
    if predictions:
        # Extracting information from the first detected object for simplicity
        result = {
            "boxes": predictions[0]['boxes'].tolist(),
            "labels": predictions[0]['labels'].tolist(),
            "scores": predictions[0]['scores'].tolist(),
        }
    else:
        result = {"message": "No objects detected"}

    return result


print("I am working dude")