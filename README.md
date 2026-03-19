# Deepfake Video Detection

A deep learning project that can tell whether a video is **real or fake (deepfake)**. Built as my Final Year B.Tech project.

---

## What does this project do?

Deepfake videos are AI-generated videos where someone's face is replaced or manipulated. They look so real that most people can't tell the difference. This project builds a system that can automatically detect them.

You upload a video → the system analyzes it → it tells you **Real** or **Fake**, and also shows a **heatmap** of which parts of the face were manipulated.

---

## How it works (simple explanation)

**Step 1 — Extract faces from the video**
The system uses MTCNN to detect and crop the face from each frame of the video. It picks 10 key frames and resizes them to 224×224.

**Step 2 — Analyze motion**
At the same time, it computes Optical Flow — this captures how the face moves between frames. Fake videos often have unnatural movement in the eyes, mouth, or cheeks that humans miss but the model catches.

**Step 3 — Understand each frame (EfficientNet-B0)**
Each face frame is passed through EfficientNet-B0 (a pretrained CNN) to extract spatial features — things like skin texture, lighting, and face structure.

**Step 4 — Understand the full video (Transformer)**
A Transformer Encoder (3 layers, 8 heads) looks at the features from all frames together, so the model understands the video as a whole, not just individual images.

**Step 5 — Combine everything**
The temporal features from the Transformer (1280-dim) and the motion features from Optical Flow (128-dim) are combined (1408-dim total) and passed through a classifier to give the final prediction.

**Step 6 — Show where the fake is (GradCAM)**
GradCAM generates a heatmap showing exactly which part of the face the model focused on — usually the eyes, mouth, or cheeks in fake videos.

---

## Model Architecture

```
Input Video
    |
    |--- MTCNN Face Extraction -------> 10 Face Frames (224x224)
    |--- Optical Flow Computation ----> Optical Flow Maps (224x224)
                                              |
              |                               |
    EfficientNet-B0 (pretrained)        Conv2D x3 + ReLU
    Classifier replaced by Identity     Channels: 2->16->32->64
              |                               |
    Transformer Encoder                 AdaptiveAvgPool2d (7x7)
    (3 layers, 8 heads, D=1280)               |
              |                         Linear 3136->128 + ReLU
    Temporal MaxPool over frames              |
              |                               |
              +------- Concat (1280 + 128) ---+
                              |
                    Linear 1408->256 + ReLU + Dropout(0.3)
                              |
                    Linear 256->1 (Output logit)
                              |
                         Real / Fake
                         + GradCAM Heatmap
                              |
                    FastAPI Web Interface
```

---

## Results

| Metric | Score |
|---|---|
| Training Accuracy | 95.36% |
| Validation Accuracy (before pruning) | 93.79% |
| Validation Loss (before pruning) | 17.76% |
| Validation Accuracy (after pruning) | 84.80% |
| Parameters (before pruning) | 40,223,629 |
| Parameters (after pruning) | 32,178,903 |
| Inference time (before optimization) | ~5 seconds per video |
| Inference time (after ONNX + pruning) | ~2 seconds per video |
| Epochs | 15 |
| Dataset | CelebDF-V2 (1,000 videos) |

---

## Limitations & Challenges

These are real challenges faced during the project — documented honestly for transparency.

**1. Dataset Diversity**
Only 1,000 videos were used due to computational constraints. The model struggles with heavily compressed videos or extreme lighting conditions not well represented in the training data. Future work will include more diverse, in-the-wild videos.

**2. Computational Cost**
The Transformer Encoder alone has 35,430,144 parameters, making real-time deployment on mid-range hardware challenging. Pruning 20% of weights across Conv2D, EfficientNet-B0, and Transformer layers reduced the model to 32,178,903 parameters and cut inference time from 5 to 2 seconds, with only a small drop in validation accuracy.

**3. GradCAM on Videos**
Applying GradCAM frame-by-frame produced inconsistent heatmaps. This was fixed by averaging heatmaps across all 10 frames, which improved consistency (average intensity in manipulated regions increased from 0.60 to 0.75). However, this occasionally smoothed over subtle single-frame artifacts.

**4. Dependency Conflicts**
facenet-pytorch required an older PyTorch version (1.9) which conflicted with the Transformer Encoder (PyTorch 1.12). This was resolved using a Docker container with pinned versions. ONNX export also required rewriting some Transformer operations to use compatible operators.

---

## Project Structure

```
hybrid-deepfake-detection/
|
|-- model.ipynb               # Full model architecture, training & evaluation (Jupyter Notebook)
|-- app.py                    # FastAPI web app + inference
|-- requirements.txt
└-- README.md
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/KhusenChettri1/hybrid-deepfake-detection.git
cd hybrid-deepfake-detection
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Download the pretrained model
Download the model file and put it in the same folder as your project:

 [Download Model from Google Drive](https://drive.google.com/file/d/1OtDJa8I5tw7cf1DxA1cn2BMitj-Bm5mR/view?usp=sharing)

### 4. To explore the model / retrain
Open the notebook in Jupyter:
```bash
jupyter notebook model.ipynb
```

### 5. Run the web app
```bash
uvicorn app:app --reload
```

Open your browser and go to `http://localhost:8000`

Upload a video and get the result instantly.

---

## Dataset

Trained on a subset of **CelebDF-V2** — one of the most challenging deepfake benchmarks available.

| | Count |
|---|---|
| Real Videos | 500 |
| Fake Videos | 500 |
| Total | 1,000 |
| Split | 80% train / 20% validation |

The dataset was balanced (500 real, 500 fake) to prevent the model from being biased toward either class.

> **Note:** The full CelebDF-V2 dataset was not used due to computational constraints. The model was trained on a representative 1,000-video subset. Download the full dataset from the official source: https://github.com/yuezunli/celeb-deepfakeforensics

---

## Pretrained Model

The model is too large to store on GitHub. Download it from Google Drive:

👉 [Download Pretrained Model](https://drive.google.com/file/d/1OtDJa8I5tw7cf1DxA1cn2BMitj-Bm5mR/view?usp=sharing)

Place the file in the same folder as `app.py` before running.

---

## Tech Stack

- Python
- PyTorch
- EfficientNet-B0
- MTCNN
- OpenCV (Optical Flow)
- Transformer Encoder
- GradCAM
- ONNX
- FastAPI

---

## About Me

**Khusen Chettri**
B.Tech in AI & Data Science — Sikkim Manipal Institute of Technology (2025)

- Email: khusengautam@gmail.com
- GitHub: [github.com/KhusenChettri1](https://github.com/KhusenChettri1)

---

If you found this useful, consider giving it a star!
