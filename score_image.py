import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import sys

# Load ONNX model
session = ort.InferenceSession("googlenet_animal_classifier.onnx")
input_name = session.get_inputs()[0].name

# Load class names from JSON file
try:
    with open('class_names.json', 'r') as f:
        class_mapping = json.load(f)
    
    # Convert to list (JSON keys are strings, need to sort by integer key)
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    print("✓ Loaded class names from class_names.json")
    
except FileNotFoundError:
    print("ERROR: class_names.json not found!")
    print("Please ensure class_names.json is in the same directory as this script.")
    sys.exit(1)

print(f"\nModel trained on {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")
print()

TARGET_SIZE = (128, 128)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(TARGET_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Get image path
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "./rhino.jpeg"

# Preprocess and predict
img_array = preprocess_image(image_path)
logits = session.run(None, {input_name: img_array})[0]
probabilities = softmax(logits[0])

# Get prediction
predicted_class = np.argmax(probabilities)
predicted_animal = class_names[predicted_class]
confidence = probabilities[predicted_class]

# Display results
print(f"Image: {image_path}")
print(f"Predicted: {predicted_animal}")
print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

print(f"\nTop 3 predictions:")
top_3_idx = np.argsort(probabilities)[-3:][::-1]
for idx in top_3_idx:
    print(f"  {class_names[idx]}: {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)")