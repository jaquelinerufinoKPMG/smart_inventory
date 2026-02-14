from ultralytics import YOLO

#TREINAMENTO DAS VACA INTEIRA COMPLEMENTANDO COM A VACA CORTADA
#model = YOLO("runs/detect/train3/weights/last.pt")

#TREINAMENTO DAS VACA INTEIRA COM O MODELO yolo26l
model = YOLO("runs/detect/train5/weights/last.pt")

model.train(resume=True)
