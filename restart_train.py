from ultralytics import YOLO

#TREINAMENTO DAS VACA INTEIRA COM O MODELO yolo26l
model = YOLO("runs/detect/train7/weights/last.pt")

model.train(resume=True)
