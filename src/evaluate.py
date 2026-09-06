from ultralytics import YOLO
import time
import torch
import cv2

MODEL_PATH = 'runs/train/hardhat_v14/weights/best.pt'
IGNORE_CLASSES = {'person'}   # we filter this out everywhere

model = YOLO(MODEL_PATH)

metrics = model.val(data='Hard-Hat-Detection-1/data.yaml', split='test',
                     imgsz=640, conf=0.25, iou=0.60,
                     plots=True, save_json=True)

print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"Precision    : {metrics.box.mp:.4f}")
print(f"Recall       : {metrics.box.mr:.4f}")
print()
print("Per-class results:")
for i, (p, r, ap) in enumerate(zip(metrics.box.p, metrics.box.r, metrics.box.ap50)):
    name = model.names[i]
    flag = "  (ignored in app)" if name in IGNORE_CLASSES else ""
    print(f"  {name:12s}: P={p:.3f} R={r:.3f} AP50={ap:.3f}{flag}")

# Speed benchmark
img = cv2.imread('Hard-Hat-Detection-1/test/images/' +
                  __import__('os').listdir('Hard-Hat-Detection-1/test/images')[0])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for _ in range(3):
    model(img, device=device, verbose=False)
t0 = time.time()
for _ in range(20):
    model(img, device=device, verbose=False)
ms = (time.time() - t0) * 1000 / 20
print(f"\nLatency: {ms:.1f}ms | FPS: {1000/ms:.1f}")