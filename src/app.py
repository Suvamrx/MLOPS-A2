"""
FastAPI inference API for Cats vs Dogs classifier
Endpoints:
- /health: GET, returns service status
- /predict: POST, accepts image file, returns class probabilities/label
"""

import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import time

request_count = 0
total_latency = 0.0

MODEL_PATH = 'models/cnn.pt'
IMG_SIZE = (224, 224)


app = FastAPI()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'request_count': request_count,
        'avg_latency': total_latency / request_count if request_count else 0.0
    }


@app.post('/predict')
async def predict(request: Request, file: UploadFile = File(...)):
    global request_count, total_latency
    start_time = time.time()
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            label = 'cat' if probs[0] > probs[1] else 'dog'
        latency = time.time() - start_time
        request_count += 1
        total_latency += latency
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {request.client.host} POST /predict latency={latency:.3f}s label={label}")
        return JSONResponse({
            'label': label,
            'probabilities': {'cat': float(probs[0]), 'dog': float(probs[1])},
            'latency': latency
        })
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {str(e)}")
        return JSONResponse({'error': str(e)}, status_code=400)
