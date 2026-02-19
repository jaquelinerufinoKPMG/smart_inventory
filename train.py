import os

from dotenv import load_dotenv
from pathlib import Path
from ultralytics import YOLO

from src.azure import downloadData

load_dotenv()

# ========= CONFIG =========
CONTAINER = "blobarcelor"

# Onde salvar localmente
LOCAL_DATASET_DIR = Path("datasets/vaca")

# Paths locais que o YOLO vai usar
DATA_YAML = Path.joinpath(LOCAL_DATASET_DIR, "data.yaml")

# ========= 1) Baixa dataset do Blob para local =========
downloadData(container_name=CONTAINER).download_files(
    prefix="vaca", local_root=LOCAL_DATASET_DIR
)

# ========= 2) Treina normalmente =========
model = YOLO("yolo26l.pt")

model.train(
    data=DATA_YAML,
    workers=0,
    epochs=500,
    imgsz=640,
    lr0=0.001,
    patience=20,
    conf=0.6,
)
