import numpy as np
import cv2
import gradio as gr
from PIL import Image
from ultralytics import YOLO

# Load your pre-trained model
model = YOLO("best.pt")

classes = ["Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf", 
           "Bell_pepper leaf spot", "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot", 
           "Corn leaf blight", "Corn rust leaf", "Peach leaf", "Potato leaf", 
           "Potato leaf early blight", "Potato leaf late blight", "Raspberry leaf", 
           "Soyabean leaf", "Soybean leaf", "Squash Powdery mildew leaf", 
           "Strawberry leaf", "Tomato Early blight leaf", "Tomato Septoria leaf spot", 
           "Tomato leaf", "Tomato leaf bacterial spot", "Tomato leaf late blight", 
           "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato mold leaf", 
           "Tomato two spotted spider mites leaf", "grape leaf", "grape leaf black rot"]

# Define the prediction function that takes an image as input and returns the predicted classes and annotated image
def predict_image(img):
    # Convert PIL image to numpy array
    img_np = np.array(img)
    
    # Perform inference with YOLO model
    results = model.predict(img_np)
    
    # Process predictions
    formatted_predictions = []
    for pred in results:
        class_probabilities = {classes[int(cls)]: float(conf) for cls, conf in zip(pred.boxes.cls, pred.boxes.conf)}
        class_name = list(class_probabilities.keys())[0]
        prob = round(class_probabilities[class_name], 2)
        formatted_class_probabilities = {class_name: prob}
        formatted_predictions.append(formatted_class_probabilities)
        # Convert box format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
        boxes = pred.boxes
        xyxy = boxes.xyxy[0].tolist()  # Get the first (and only) bounding box coordinates
        # Extract coordinates and format them as integers
        x_min, y_min, x_max, y_max = map(int, xyxy)
        
        # Draw bounding box on the image
        cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Put class label and confidence score on the image
        cv2.putText(img_np, f"{class_name} {prob}", (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert annotated image back to PIL format for display
    annotated_img_pil = Image.fromarray(img_np)
    
    return formatted_class_probabilities, annotated_img_pil

title_markdown = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <div>
    <h1 >Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams</h1>
  </div>
</div>
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <div style="display:flex; gap: 0.25rem;" align="center">
        <a href='https://github.com/iampujan/plant-disease-detection'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
    </div>
</div>
<p>Developers: Pujan Thapa, Raza Mehar, Syed Najam Mehdi</p>
<p> University of Naples Federico II </p>
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

# Create the Gradio interface
gr.Interface(
    gr.Markdown(title_markdown)
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Label(num_top_classes=15, label="Predicted Classes"),
        gr.Image(type="pil", label="Annotated Image")
    ],
    title="Plant Disease Detection using YOLO8m 🌱🦠",
    allow_flagging=False  # Disable flagging
).launch()
