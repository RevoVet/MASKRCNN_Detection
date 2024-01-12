from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw, ImageFont
import io
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import functional as F
import numpy as np
import base64
import openai
import os

from dotenv import load_dotenv
load_dotenv()  # This loads the environment variables from .env


openai.api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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



def preprocess_detections(detections):
    # Group by labels and keep the one with the highest score for each label
    max_score_per_label = {}

    for detection in detections:
        label = detection['label']
        score = detection['score']

        if label not in max_score_per_label or max_score_per_label[label]['score'] < score:
            max_score_per_label[label] = detection

    # Convert back to list and sort by score
    processed_detections = sorted(max_score_per_label.values(), key=lambda x: x['score'], reverse=True)

    return processed_detections



def generate_diagnostic_text(detections):
    # print(detections)
    detections = preprocess_detections(detections)
    print(detections)

    if not detections:
        prompt = "The pet's eye seems healthy with no detected issues. What general advice would you give to the pet owner for maintaining their pet's eye health?"
    else:
        prompt = "Based on the following pet eye diagnoses and their probabilities, provide a detailed diagnosis, explanation, and recommended next steps for each:\n"
        for det in detections:
            confidence_level = "highly likely" if det['score'] > 0.8 else "possibly"
            prompt += f"- Diagnosis: {label_dict[det['label']]} (Probability: {det['score']:.2f}, Confidence: {confidence_level})\n"
            #prompt += f"Possible treatment of {label_dict[det['label']]} and the chance of full recovery: \n"

    #prompt += "\nProvide a concluding sentence with general advice for the pet owner. "

    prompt += "\nProvide this text in German."

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  # Or other suitable engine
        prompt=prompt,
        max_tokens=450  # Adjust as needed
    )

    return response.choices[0].text.strip()



def annotate_image(image, predictions):
    # Convert tensor image back to PIL for annotation
    #image_pil = F.to_pil_image(image)
    image_uint8 = (image * 255).byte()
    scores = [score for score in predictions[0]['scores'] if score > 0.5]
    if len(scores) == 0:
        return F.to_pil_image(image_uint8), [{"label": 0, "score":0.0}]
    
    boxes = predictions[0]['boxes'][0:len(scores)]
    labels = predictions[0]['labels'][0:len(scores)]
    masks = predictions[0]['masks'][0:len(scores)] > 0.5  # Masks are assumed to be in the predictions


    image_with_boxes = draw_bounding_boxes(image_uint8, boxes, colors="red")
    image_with_masks = draw_segmentation_masks(image_with_boxes, masks.max(dim=0)[0])


    annotated_image = F.to_pil_image(image_with_masks)
    detections = []
    draw = ImageDraw.Draw(annotated_image)
    for box, label, score in zip(boxes, labels, scores):
        # Optionally, add labels and scores as text
        draw.text((box[0], box[1]), f'Label: {label_dict[label.item()]}, Score: {score:.2f}', fill="white")
        detections.append({
            "label": label.item(),  # Convert to Python scalar
            "score": round(score.item(), 2)  # Convert to Python scalar and round off
        })


    return annotated_image, detections


# @app.post("/test-upload/")
# async def test_upload(file: UploadFile = File(...)):
#     # Read image file
#     image_contents = await file.read()
#     # Preprocess the image
#     image = preprocess_image(image_contents)
#     # Get predictions
#     predictions = predict(image, model)
#     annotated_image, detections = annotate_image(image, predictions)
#     diagnostic_text = generate_diagnostic_text(detections)

#     img_byte_arr = io.BytesIO()
#     annotated_image.save(img_byte_arr, format='JPEG')
#     img_byte_arr = img_byte_arr.getvalue()

#     # Convert to base64 for easy transfer over HTTP
#     encoded_img = base64.b64encode(img_byte_arr).decode('utf-8')

#     print("Diagnose successful")
#     return {"filename": file.filename, "message": "Test upload successful"}


@app.get("/", response_class=HTMLResponse)
async def root():
    table = "<table><tr><th>Index</th><th>Label</th></tr>"
    for index, label in label_dict.items():
        table += f"<tr><td>{index}</td><td>{label}</td></tr>"
    table += "</table>"
    return f"""
        <h1>Hello Martin!</h1>
        <p>This is the Revovet API</p>
        <p>Check out the documentation under /docs </p>
        <p>The label encoding is as follows: </p>
        {table}
    """


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Read image file
    image_contents = await file.read()
    # Preprocess the image
    image = preprocess_image(image_contents)
    # Get predictions
    predictions = predict(image, model)
    annotated_image, detections = annotate_image(image, predictions)

    # Preprocess the detections
    # print(preprocess_detections(detections))

    diagnostic_text = generate_diagnostic_text(detections)

    # Save the annotated image to a bytes buffer
    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Convert to base64 for easy transfer over HTTP
    encoded_img = base64.b64encode(img_byte_arr).decode('utf-8')

    return {"filename": file.filename, 
            "annotated_image": encoded_img,
            "detections": detections, 
            "diagnostic_text": diagnostic_text}


print("I am working dude")