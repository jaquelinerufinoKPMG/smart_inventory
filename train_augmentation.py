import os

from dotenv import load_dotenv
from pathlib import Path
from ultralytics import YOLO

from src.azure import downloadData

load_dotenv()

# ========= CONFIG =========

CONTAINER = "blobarcelor"

# Diretório local do dataset
LOCAL_DATASET_DIR = Path("datasets/vaca_tilada")

# Arquivo YAML do dataset
DATA_YAML = LOCAL_DATASET_DIR / "data.yaml"

# Número de workers (CPU)
NUM_WORKERS = max(os.cpu_count() - 1, 1)

# ========= 1) Baixar dataset =========

downloadData(container_name=CONTAINER).download_files(
    prefix="vaca",
    local_root=LOCAL_DATASET_DIR
)

print("Dataset baixado com sucesso")

# ========= 2) Carregar modelo =========

model = YOLO("yolo26l.pt")

# ========= 3) Treinamento otimizado =========

model.train(

    # dataset
    data=DATA_YAML,

    # treino
    epochs=300,
    imgsz=640,

    # performance
    batch=-1,
    workers=NUM_WORKERS,
    device=0,
    cache=True,
    amp=True,

    # otimização
    lr0=0.001,
    patience=30,

    # ========= DATA AUGMENTATION =========

    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,

    fliplr=0.5,
    flipud=0.2,

    mosaic=1.0,
    mixup=0.2,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    # ========= REGULARIZAÇÃO =========

    erasing=0.2,
    perspective=0.0005,

)
