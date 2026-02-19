from ultralytics import YOLO
import pandas as pd
from pathlib import Path

validations = []
metrics_result = {}

models_path = Path(r"src\yolo models")
yaml_path = Path(r"datasets\vaca\data.yaml")
test_path = Path(r"datasets\teste")
output_path = Path(r"data\output")

THRESHOLD = {"Default":0.25, 
             "50%":0.50,
             "75%":0.75,
             "85%":0.85,}

for file in models_path.rglob("*"):
    print(f"Processando arquivo: {file}")
    model_path = Path(file)
    model = YOLO(model_path)    

    metrics = model.val(data=yaml_path)
    for key, value in THRESHOLD.items():
        metrics_result = {}
        total = 0
        print(f"Processando predict para threshold {key}")
        results = model.predict(
            source=test_path,
            save=True,
            show_labels=True,
            show_conf=True,
            stream=True,
            conf=value,
        )

        for r in results:
            total += len(r.boxes)
        
        metrics_result.update({"Nome":model_path.name.split(".")[0],
                               "Modelo":model.model.yaml.get("yaml_file"),
                               "Epochs": model.ckpt['train_args'].get("epochs"),
                               "mAP50": metrics.box.map50.round(3),
                               "mAP50-95": metrics.box.map.round(3),
                               "Precision":metrics.box.mp.round(3),
                               "Recall": metrics.box.mr.round(3),
                               "Threshold":key,
                               "Total Manual": 3269,
                               "Total Predict": total,})

        validations.append(metrics_result)

export_validations = pd.DataFrame(validations)


output_path = rf"{output_path}\train_analysis_data.xlsx"
export_validations.to_excel(output_path)
print(f"Arquivo salvo em {output_path}")