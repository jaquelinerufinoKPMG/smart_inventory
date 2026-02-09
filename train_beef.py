import os
from pathlib import Path
from ultralytics import YOLO

# 1) Base folder opcional (se não existir, usa pasta atual)
folder_path = os.getenv("YOLO_FOLDER", ".")
folder_path = Path(folder_path)

# 2) Dataset (melhor com caminho absoluto pra evitar confusão de Jupyter)
data_yaml = (folder_path / "datasets" / "vaca_tilada" / "data.yaml").resolve()

# 3) Escolha UMA abordagem:
#    - Transfer learning (recomendado): carregar .pt pretreinado
weights_path = (folder_path / "yolo26n.pt").resolve()
model = YOLO(str(weights_path))

# 4) Treino com pasta de saída fixa + checkpoint frequente
model.train(
    data=str(data_yaml),
    epochs=120,
    imgsz=512,
    batch=1,
    workers=0,
    cache=False
)
