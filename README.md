# 🦺 Hard Hat Safety Detection — YOLOv8

Real-time object detection system that identifies whether workers are wearing
safety helmets, built with a custom fine-tuned YOLOv8 model.

## 🎯 Project Overview

This project fine-tunes YOLOv8 on a custom construction-site dataset to detect
`head` (no helmet — safety violation) and `helmet` (compliant) in images and
video footage. It includes a FastAPI endpoint for image-based detection and a
script for processing video files with an automatic safety-violation overlay.

## 📊 Results

| Class  | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|--------|-----------|--------|---------|--------------|
| head   | 0.944     | 0.930  | 0.958   | 0.699        |
| helmet | 0.962     | 0.937  | 0.975   | 0.717        |

- Trained on 6,000+ annotated images (Roboflow "Hard-Hat-Detection" dataset)
- 20 epochs, YOLOv8s, trained on CPU (~14.6 hours)
- Inference speed: ~155ms/frame (6.5 FPS) on CPU

## 🛠️ Tech Stack

- **Model**: YOLOv8s (Ultralytics)
- **Training**: PyTorch, custom dataset via Roboflow
- **API**: FastAPI
- **Video/Image processing**: OpenCV

## 📁 Project Structure

\```
object_detection/
├── src/
│   ├── train.py           # Fine-tuning script
│   ├── evaluate.py        # Model evaluation (mAP, precision, recall, FPS)
│   ├── verify_dataset.py  # Annotation visualization
│   ├── infer_video.py     # Video file inference with safety alerts
│   └── api.py             # FastAPI /detect endpoint
├── requirements.txt
└── README.md
\```

## 🚀 How to Run

1. Clone the repo and install dependencies:
   \```
   pip install -r requirements.txt
   \```
2. Download the dataset separately (see Dataset section) or use your own,
   matching the `data.yaml` format.
3. Run inference on a video:
   \```
   python src/infer_video.py
   \```
4. Or start the API:
   \```
   uvicorn src.api:app --reload --port 8000
   \```
   Then visit `http://localhost:8000/docs` to test the `/detect` endpoint.

## 📦 Dataset

Dataset sourced from [Roboflow Universe — Hard Hat Detection](https://universe.roboflow.com/computer-vision-filqz/hard-hat-detection-62rrp).
Not included in this repo due to size — download separately and place in the
project root as `Hard-Hat-Detection-1/`.

## 🔍 Known Limitations

- Trained on CPU due to hardware constraints, limiting real-time FPS
- Original dataset included a `person` class with insufficient samples (142
  instances vs 4,800+ for helmet); this class is filtered out in the
  application logic rather than removed from training

## 📈 Future Improvements

- Retrain on GPU (Colab) for faster iteration and higher epoch counts
- Add authentication to the API
- Dockerize for cloud deployment
- Real-time webcam optimization (frame skipping, smaller inference size)