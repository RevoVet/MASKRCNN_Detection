from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageDraw, ImageFont
import io
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import functional as F
import numpy as np
import base64




app = FastAPI()

# label_dict = {1:"Vaskularisation"}
label_dict = {i+1:name for i, name in enumerate(["Linsentruebung", "Vorderkammertruebung", "Vaskularisation", "Hornhauttruebung", "Hornhautpigmentierung", "Hornhautdefekt"])}


def load_model():
    # placeholder function
    model = torch.load("revovet_general_initial_model_epoch_9.pt", map_location=torch.device("cpu"))
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



def annotate_image(image, predictions):
    # Convert tensor image back to PIL for annotation
    #image_pil = F.to_pil_image(image)
    image_uint8 = (image * 255).byte()

    scores = [score for score in predictions[0]['scores'] if score > 0.5]
    boxes = predictions[0]['boxes'][0:len(scores)]
    labels = predictions[0]['labels'][0:len(scores)]
    masks = predictions[0]['masks'][0:len(scores)] > 0.5  # Masks are assumed to be in the predictions


    image_with_boxes = draw_bounding_boxes(image_uint8, boxes, colors="red")
    image_with_masks = draw_segmentation_masks(image_with_boxes, masks.max(dim=0)[0], alpha=0.7)


    annotated_image = F.to_pil_image(image_with_masks)

    draw = ImageDraw.Draw(annotated_image)
    for box, label, score in zip(boxes, labels, scores):
        # Optionally, add labels and scores as text
        draw.text((box[0], box[1]), f'Label: {label_dict[label.item()]}, Score: {score:.2f}', fill="white")

    return annotated_image




@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Read image file
    image_contents = await file.read()
    
    # Preprocess the image
    image = preprocess_image(image_contents)

    # Get predictions
    predictions = predict(image, model)

    annotated_image = annotate_image(image, predictions)

    # Save the annotated image to a bytes buffer
    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Convert to base64 for easy transfer over HTTP
    encoded_img = base64.b64encode(img_byte_arr).decode('utf-8')

    return {"filename": file.filename, "annotated_image": encoded_img}


print("I am working dude")