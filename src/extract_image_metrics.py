from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import cv2
import numpy as np

@dataclass
class ImgMetrics:
    file: str
    w: int
    h: int
    brightness_mean: float
    brightness_std: float
    contrast_std: float
    sharpness_lap_var: float
    pct_black: float
    pct_white: float
    entropy: float

def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def extract_metrics(image_path: Path) -> ImgMetrics:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Não consegui ler imagem: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness_mean = float(np.mean(gray))
    brightness_std  = float(np.std(gray))
    contrast_std    = brightness_std  # mesma coisa aqui (std do gray)
    sharpness       = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # “black crush” e “white clip”
    pct_black = float((gray <= 5).mean() * 100.0)
    pct_white = float((gray >= 250).mean() * 100.0)

    ent = _entropy(gray)

    return ImgMetrics(
        file=str(image_path),
        w=w, h=h,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        contrast_std=contrast_std,
        sharpness_lap_var=sharpness,
        pct_black=pct_black,
        pct_white=pct_white,
        entropy=ent,
    )
