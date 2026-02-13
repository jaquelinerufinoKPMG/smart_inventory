from ultralytics import YOLO

model_path = fr"runs\detect\train2\weights\best.pt"
yaml_path = fr"datasets\vaca_tilada\data.yaml"

model = YOLO(model_path)

model.train(
    data=yaml_path,
    workers=0,
    epochs=160,
    imgsz=640,
    lr0=0.001,
    patience=20, 
    conf=0.6, 
)
