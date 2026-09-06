from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model('https://ultralytics.com/images/bus.jpg')
results[0].show()

for box in results[0].boxes:
    print(f"{model.names[int(box.cls)]}: {float(box.conf):.2f}")