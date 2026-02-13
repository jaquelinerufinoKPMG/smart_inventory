from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/last.pt")
model.train(resume=True)
