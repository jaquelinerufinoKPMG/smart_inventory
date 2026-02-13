from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/last.pt")
model.train(resume=True)
