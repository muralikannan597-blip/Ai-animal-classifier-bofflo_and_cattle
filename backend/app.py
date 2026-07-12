from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

import os

# Point Flask to the frontend directory so it can serve the UI!
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='/')

@app.route('/')
def serve_frontend():
    return app.send_static_file('index.html')

CORS(app) # Allows your frontend HTML to communicate with this backend

print("Loading AI Model...")

# 1. Setup the model architecture (Exactly the same as we did in train.py)
device = torch.device("cpu") # Web servers usually run inference on CPU
model = models.resnet18(weights=None) 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) # 2 classes: buffalo and cattle

# 2. Load your newly trained "brain" into the architecture
model.load_state_dict(torch.load('livestock_model.pth', map_location=device, weights_only=True))
model.eval() # Lock the model into "evaluation" mode so it doesn't try to learn

class_names = ['buffalo', 'cattle'] # The order matches your training data

# 3. Define the rules for incoming images (matches 'val' from dp.py)
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Create the API endpoint that listens for images
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Read the uploaded image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Apply the transforms and prep it for the AI
        tensor = image_transform(img).unsqueeze(0).to(device)
        
        # Pass the image to the AI and get the guess
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            
        result = class_names[predicted.item()]
        
        # Send the answer back to the user
        return jsonify({'prediction': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server is starting! Waiting for images...")
    app.run(debug=True, port=5000)
