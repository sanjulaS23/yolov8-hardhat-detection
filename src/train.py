from ultralytics import YOLO
import torch

def train(data_yaml='Hard-Hat-Detection-1/data.yaml',
          model_size='s',
          epochs=20,
          img_size=640,
          batch_size=8,
          name='hardhat_v1'):

    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model = YOLO(f'yolov8{model_size}.pt')

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project='runs/train',
        name=name,
        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        patience=20,
        save=True,
        save_period=10,
        cache=True,
        workers=4,
        plots=True,
        verbose=True
    )

if __name__ == '__main__':
    train()