import torch
import torch.nn as nn
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
import logging
import os
import base64
import io
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torch.nn import TransformerEncoder, TransformerEncoderLayer  # Added missing imports

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Debug: Print current directory and files
print(f"Current directory: {os.getcwd()}")
print(f"Files: {os.listdir()}")

# Load ONNX model
try:
    session = ort.InferenceSession("hybrid_model.onnx")  # Updated to match existing file
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    dummy_input = np.random.randn(1, 3, 3, 224, 224).astype(np.float32)
    dummy_flow = np.random.randn(1, 3, 2, 224, 224).astype(np.float32)
    inputs = {input_names[0]: dummy_input, input_names[1]: dummy_flow}
    outputs = session.run(output_names, inputs)
    print(f"ONNX inference successful, output shape: {outputs[0].shape}")
except Exception as e:
    print(f"ONNX inference failed: {str(e)}")

# Transforms
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MTCNN for face detection
mtcnn = MTCNN(image_size=224, margin=30, keep_all=False, min_face_size=50, device='cpu')

# Define the HybridModel
class HybridModel(nn.Module):
    def __init__(self, num_frames=3):
        super(HybridModel, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        self.efficientnet.classifier = nn.Identity()

        self.feature_dim = 1280
        self.num_frames = num_frames

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, self.feature_dim))

        encoder_layer = TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.2
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        # Optical flow branch
        self.flow_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.flow_fc = nn.Linear(64 * 7 * 7, 128)

        # Fusion + classification
        self.fc1 = nn.Linear(self.feature_dim + 128, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, frames, flows):
        batch_size, num_frames, c, h, w = frames.size()

        # Frame feature extraction
        frames = frames.view(batch_size * num_frames, c, h, w)
        frame_features = self.efficientnet(frames)
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # Add positional encoding
        frame_features = frame_features + self.pos_embedding[:, :num_frames, :]

        # Temporal modeling
        transformer_out = self.transformer(frame_features)
        temporal_features = transformer_out.max(dim=1)[0]

        # Optical flow feature extraction
        flow_features = []
        for t in range(num_frames):
            flow_map = flows[:, t, :, :, :]  # [B, 2, H, W]
            conv_out = self.flow_conv(flow_map)
            flow_features.append(conv_out)

        flow_features = torch.stack(flow_features, dim=1)  # [B, T, 64, 7, 7]
        flow_features = flow_features.max(dim=1)[0]        # [B, 64, 7, 7]
        flow_features = flow_features.view(batch_size, -1) # [B, 3136]
        flow_features = self.relu(self.flow_fc(flow_features))  # [B, 128]

        # Combine and classify
        combined = torch.cat([temporal_features, flow_features], dim=1)
        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.fc2(x)

        return x, temporal_features

# Grad-CAM Model Wrapper
class GradCAMModel(nn.Module):
    def __init__(self, model):
        super(GradCAMModel, self).__init__()
        self.model = model
        self.target_layer = model.efficientnet.features[-1]

    def forward(self, frames, flows):
        batch_size, num_frames, c, h, w = frames.size()
        frames = frames.view(batch_size * num_frames, c, h, w)
        x = self.model.efficientnet.features(frames)
        x = x.view(batch_size, num_frames, -1)
        x = self.model.transformer(x)
        temporal_features = x.max(dim=1)[0]
        flow_features = []
        for t in range(num_frames):
            flow_map = flows[:, t, :, :, :]
            conv_out = self.model.flow_conv(flow_map)
            flow_features.append(conv_out)
        flow_features = torch.stack(flow_features, dim=1).max(dim=1)[0].view(batch_size, -1)
        flow_features = self.model.relu(self.model.flow_fc(flow_features))
        combined = torch.cat([temporal_features, flow_features], dim=1)
        x = self.model.dropout(self.model.relu(self.model.fc1(combined)))
        x = self.model.fc2(x)
        return x

# Load PyTorch model
device = 'cpu'
model = HybridModel(num_frames=3).to(device)
try:
    model.load_state_dict(torch.load("deepfake_transformer_pruned (1).pth", map_location=device))  # Updated to match existing file
    model.eval()
    print("PyTorch model loaded successfully")
except Exception as e:
    print(f"PyTorch model loading failed: {str(e)}")

# Initialize Grad-CAM
cam_model = GradCAMModel(model)
cam = GradCAM(cam_model, target_layer=cam_model.target_layer)

def extract_face(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(frame_rgb)
    if boxes is not None and len(boxes) > 0:
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        max_area_idx = np.argmax(areas)
        box = boxes[max_area_idx]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.resize(face, (224, 224))
    return np.zeros((224, 224, 3), dtype=np.uint8)

def compute_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.GaussianBlur(flow, (5, 5), 0)
    return cv2.resize(flow, (224, 224))

def preprocess_video(video_file, num_frames=3):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail="Failed to open video. Please ensure the file is a valid video (e.g., .mp4) and not corrupted.")
    
    frames = []
    flows = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames + 1, dtype=int)[:num_frames]
    logging.info(f"Total frames: {total_frames}, Selected indices: {frame_indices}")

    prev_frame = None
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        face = extract_face(frame)
        frames.append(face)
        if prev_frame is not None:
            flow = compute_optical_flow(prev_frame, face)
            flows.append(flow)
        else:
            flows.append(np.zeros((224, 224, 2), dtype=np.float32))
        prev_frame = face

    cap.release()
    if len(frames) < num_frames:
        frames.extend([np.zeros((224, 224, 3), dtype=np.uint8)] * (num_frames - len(frames)))
        flows.extend([np.zeros((224, 224, 2), dtype=np.float32)] * (num_frames - len(flows)))

    frames = np.stack(frames)
    flows = np.stack(flows)
    frames = torch.stack([val_test_transform(Image.fromarray(f)) for f in frames])
    flows = torch.tensor(flows, dtype=torch.float32).permute(0, 3, 1, 2)
    return frames.unsqueeze(0), flows.unsqueeze(0)

def analyze_gradcam(cam_map):
    """Analyze Grad-CAM heatmap to generate a textual description."""
    if cam_map.ndim != 2:
        cam_map = np.mean(cam_map, axis=0)
    cam_map = np.interp(cam_map, (cam_map.min(), cam_map.max()), (0, 1))
    
    # Divide the image into regions (e.g., 3x3 grid)
    h, w = cam_map.shape
    region_h, region_w = h // 3, w // 3
    regions = {
        'top-left': cam_map[:region_h, :region_w],
        'top-center': cam_map[:region_h, region_w:2*region_w],
        'top-right': cam_map[:region_h, 2*region_w:],
        'middle-left': cam_map[region_h:2*region_h, :region_w],
        'middle-center': cam_map[region_h:2*region_h, region_w:2*region_w],
        'middle-right': cam_map[region_h:2*region_h, 2*region_w:],
        'bottom-left': cam_map[2*region_h:, :region_w],
        'bottom-center': cam_map[2*region_h:, region_w:2*region_w],
        'bottom-right': cam_map[2*region_h:, 2*region_w:]
    }
    
    # Calculate mean activation per region
    region_activations = {k: np.mean(v) for k, v in regions.items()}
    max_region = max(region_activations, key=region_activations.get)
    max_activation = region_activations[max_region]
    
    # Generate description
    if max_activation > 0.5:  # Threshold for significant activation
        description = f"The model focused strongly on the {max_region.replace('-', ' ')} region of the face, indicating this area was critical for the prediction."
    else:
        description = "The model showed distributed attention across the face, with no single region dominating the prediction."
    
    return description

def generate_gradcam(model, cam, frames, flows, device):
    model.eval()
    frames, flows = frames.to(device), flows.to(device)
    
    model.zero_grad()
    output, _ = model(frames, flows)
    output.backward(retain_graph=True)
    cam_map = cam(scores=output, class_idx=0)
    cam_map = cam_map[0]
    cam_map = cam_map.cpu().detach().numpy()
    
    # Analyze Grad-CAM for description
    gradcam_description = analyze_gradcam(cam_map)
    
    if cam_map.ndim > 2:
        cam_map = np.mean(cam_map, axis=0)
    if cam_map.ndim != 2:
        raise ValueError(f"Expected 2D array for cam_map, got shape {cam_map.shape}")
    cam_map = np.interp(cam_map, (cam_map.min(), cam_map.max()), (0, 255)).astype(np.uint8)
    cam_map = cv2.resize(cam_map, (224, 224))
    heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    frame_tensor = frames[0, 0].cpu() * torch.tensor(std) + torch.tensor(mean)
    frame = cv2.cvtColor((frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap * 0.4 + frame / 255.0
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    # Convert to base64
    superimposed_img = (superimposed_img * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', superimposed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64, output, gradcam_description

def generate_prediction(frames, flows):
    model.eval()
    output, _ = model(frames, flows)
    prob_fake = torch.sigmoid(output).item()
    prediction = "Fake" if prob_fake > 0.1 else "Real"
    confidence = 100 * (1 - prob_fake) if prob_fake <= 0.5 else 100 * prob_fake

    # Verify with ONNX
    frames_np = frames.cpu().numpy()
    flows_np = flows.cpu().numpy()
    inputs = {input_names[0]: frames_np, input_names[1]: flows_np}
    outputs = session.run(output_names, inputs)[0]
    onnx_prob_fake = torch.sigmoid(torch.tensor(outputs)).item()
    onnx_prediction = "Fake" if onnx_prob_fake > 0.1 else "Real"
    onnx_confidence = 100 * (1 - onnx_prob_fake) if onnx_prob_fake <= 0.5 else 100 * onnx_prob_fake
    print(f"PyTorch Prediction: {prediction}, Confidence: {confidence}%")
    print(f"ONNX Prediction: {onnx_prediction}, Confidence: {onnx_confidence}%")

    # Generate Grad-CAM
    heatmap_base64, _, gradcam_description = generate_gradcam(model, cam, frames, flows, device)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "heatmap": f"data:image/png;base64,{heatmap_base64}",
        "gradcam_description": gradcam_description
    }

@app.post("/predict/")
async def predict(video: UploadFile = File(...)):
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(video.file.read())
        frames, flows = preprocess_video("temp_video.mp4", num_frames=3)
        result = generate_prediction(frames, flows)
        return JSONResponse(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deepfake Detector</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            body {
                background: linear-gradient(135deg, #4A90E2, #F5F7FA);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
                width: 100%;
                max-width: 600px;
                text-align: center;
                position: relative;
            }
            .header {
                margin-bottom: 30px;
            }
            .header h1 {
                color: #2C3E50;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .upload-area {
                width: 300px;
                height: 300px;
                border: 3px dashed #3498DB;
                border-radius: 50%;
                margin: 0 auto 30px;
                display: flex;
                justify-content: center;
                align-items: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                background: #ECF0F1;
                border-color: #2980B9;
            }
            .upload-area.invalid {
                border-color: #E74C3C;
            }
            .upload-label {
                color: #3498DB;
                font-size: 1.3em;
                font-weight: 700;
            }
            .upload-icon {
                font-size: 2em;
                margin-bottom: 10px;
            }
            #videoInput {
                display: none;
            }
            .loading {
                display: none;
                width: 60px;
                height: 60px;
                border: 6px solid #F3F3F3;
                border-top: 6px solid #3498DB;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 30px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background: #F9F9F9;
                border-radius: 15px;
                display: none;
                animation: fadeIn 0.5s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .result p {
                margin: 15px 0;
                color: #2C3E50;
                font-size: 1.2em;
            }
            .result .prediction.real { color: #27AE60; font-weight: 700; font-size: 1.5em; }
            .result .prediction.fake { color: #E74C3C; font-weight: 700; font-size: 1.5em; }
            .heatmap {
                margin-top: 20px;
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            .gradcam-description {
                margin-top: 15px;
                font-size: 1.1em;
                color: #34495E;
            }
            .error {
                color: #E74C3C;
                margin-top: 20px;
                display: none;
                padding: 10px;
                background: #FFEDEE;
                border-radius: 10px;
                position: relative;
            }
            .error .close {
                position: absolute;
                right: 10px;
                top: 5px;
                cursor: pointer;
                font-size: 1.2em;
            }
            .help-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: #3498DB;
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                font-size: 1.2em;
                cursor: pointer;
                transition: background 0.3s;
            }
            .help-btn:hover { background: #2980B9; }
            @media (max-width: 600px) {
                .container { padding: 20px; }
                .upload-area { width: 200px; height: 200px; }
                .header h1 { font-size: 2em; }
                .result p { font-size: 1em; }
                .heatmap { max-width: 200px; }
                .gradcam-description { font-size: 0.9em; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Deepfake Detector</h1>
            </div>
            <div class="upload-area" id="uploadArea">
                <div>
                    <i class="fas fa-upload upload-icon"></i>
                    <label for="videoInput" class="upload-label">Drag & Drop a Video or Click to Upload (.mp4)</label>
                    <input type="file" id="videoInput" name="video" accept="video/mp4" onchange="handleUpload(event)">
                </div>
            </div>
            <div class="loading" id="loading"></div>
            <div class="result" id="result">
                <p><span class="prediction" id="prediction"></span></p>
                <p><strong>Confidence:</strong> <span id="confidence"></span>%</p>
                <img id="heatmap" class="heatmap" src="" alt="Grad-CAM Heatmap" style="display: none;">
                <p class="gradcam-description" id="gradcamDescription"></p>
            </div>
            <div class="error" id="error"><span class="close" onclick="this.parentElement.style.display='none'">×</span><span id="errorMessage"></span></div>
            <button class="help-btn" onclick="alert('Help section coming soon!')">?</button>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const videoInput = document.getElementById('videoInput');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const prediction = document.getElementById('prediction');
            const confidence = document.getElementById('confidence');
            const heatmap = document.getElementById('heatmap');
            const gradcamDescription = document.getElementById('gradcamDescription');
            const error = document.getElementById('error');
            const errorMessage = document.getElementById('errorMessage');

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.background = '#ECF0F1';
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.background = 'transparent';
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.background = 'transparent';
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('video/')) {
                    handleFile(file);
                } else {
                    uploadArea.classList.add('invalid');
                    showError('Please upload a valid .mp4 file');
                    setTimeout(() => uploadArea.classList.remove('invalid'), 2000);
                }
            });

            function handleUpload(event) {
                const file = event.target.files[0];
                if (file && file.type.startsWith('video/')) {
                    handleFile(file);
                } else {
                    uploadArea.classList.add('invalid');
                    showError('Please upload a valid .mp4 file');
                    setTimeout(() => uploadArea.classList.remove('invalid'), 2000);
                }
            }

            function handleFile(file) {
                loading.style.display = 'block';
                result.style.display = 'none';
                error.style.display = 'none';
                heatmap.style.display = 'none';
                gradcamDescription.textContent = '';

                const formData = new FormData();
                formData.append('video', file);

                fetch('/predict/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) throw new Error('Prediction failed');
                    return response.json();
                })
                .then(data => {
                    if (!data.prediction || data.confidence === undefined || !data.heatmap || !data.gradcam_description) {
                        throw new Error('Invalid response from server');
                    }
                    prediction.textContent = data.prediction;
                    prediction.className = 'prediction ' + data.prediction.toLowerCase();
                    confidence.textContent = data.confidence.toFixed(2);
                    heatmap.src = data.heatmap;
                    heatmap.style.display = 'block';
                    gradcamDescription.textContent = data.gradcam_description;
                    result.style.display = 'block';
                })
                .catch(err => {
                    showError(err.message || 'An error occurred during prediction');
                })
                .finally(() => {
                    loading.style.display = 'none';
                });
            }

            function showError(message) {
                errorMessage.textContent = message;
                error.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)