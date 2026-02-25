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

def score_against_profile(m: ImgMetrics, profile: dict) -> tuple[float, dict]:
    feats = profile["features"]
    z = {}
    for k, stats in feats.items():
        val = getattr(m, k)
        z[k] = (val - stats["mean"]) / stats["std"]

    # score global (L2 dos z-scores)
    score = float(np.sqrt(np.sum(np.array(list(z.values())) ** 2)))
    return score, z

def classify(m: ImgMetrics, profile: dict) -> tuple[str, float, dict]:
    score, z = score_against_profile(m, profile)

    # guardrails simples baseados em percentis do dataset original
    feats = profile["features"]
    hard_fail = []
    warn = []

    # brilho fora do miolo (p05-p95) vira warning; muito fora vira fail
    b = m.brightness_mean
    if b < feats["brightness_mean"]["p05"] or b > feats["brightness_mean"]["p95"]:
        warn.append("brightness_outside_p05_p95")

    # nitidez muito baixa (abaixo p05) costuma ser problema real
    if m.sharpness_lap_var < feats["sharpness_lap_var"]["p05"]:
        hard_fail.append("too_blurry")

    # muito estourado / muito preto
    if m.pct_white > 5.0:
        warn.append("too_much_white_clip")
    if m.pct_black > 5.0:
        warn.append("too_much_black_crush")

    # score de distância global
    # (calibrável: ~3-5 costuma ser “ok”, >7-9 costuma ser “longe”, dependendo do dataset)
    if score >= 9.0:
        hard_fail.append("too_far_global")
    elif score >= 6.0:
        warn.append("far_global")

    if hard_fail:
        return "FAIL", score, {"hard_fail": hard_fail, "warn": warn, "z": z}
    if warn:
        return "WARN", score, {"hard_fail": hard_fail, "warn": warn, "z": z}
    return "PASS", score, {"hard_fail": hard_fail, "warn": warn, "z": z}