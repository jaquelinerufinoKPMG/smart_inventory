from src.extract_image_metrics import extract_metrics
import numpy as np

from dataclasses import asdict

from pathlib import Path
import json

TRAIN_DATASET = Path(r"datasets\vaca\images")
OUTPUT_JSON = Path(r"data\output\profile.json")

def build_profile(image_folder: Path, out_json: Path) -> dict:
    metrics = []
    for p in image_folder.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            metrics.append(extract_metrics(p))

    # agrega estatísticas (média e desvio) por feature
    keys = [k for k in asdict(metrics[0]).keys() if k not in {"file", "w", "h"}]
    profile = {"features": {}, "n_images": len(metrics)}

    for k in keys:
        arr = np.array([getattr(m, k) for m in metrics], dtype=float)
        profile["features"][k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) + 1e-9),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
        }

    out_json.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile
