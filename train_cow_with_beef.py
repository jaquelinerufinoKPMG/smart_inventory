import os

from ultralytics import YOLO

model_path = os.path.join("runs","detect","train2","weights","best.pt")
yaml_path = os.path.join("datasets","vaca_tilada","data.yaml")

model = YOLO(model_path)

model.train(
    data=yaml_path,
    epochs=160,
    imgsz=640,
    batch=16,
    workers=0,
    optimizer="SGD",
    lr0=0.001,
    momentum=0.937,
    mosaic=0.2,
    mixup=0.0,
)