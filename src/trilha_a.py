from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Det:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def iter_tiles(img: np.ndarray, tile: int = 640, overlap: float = 0.25):
    """
    Gera crops (tiles) com overlap.
    Retorna: (crop, x0, y0) onde (x0,y0) é o offset do crop na imagem original.
    """
    h, w = img.shape[:2]
    stride = max(1, int(tile * (1 - overlap)))

    # garante que o último tile encoste no final da imagem
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if xs[-1] != w - tile:
        xs.append(w - tile)
    if ys[-1] != h - tile:
        ys.append(h - tile)

    for y0 in ys:
        for x0 in xs:
            crop = img[y0:y0 + tile, x0:x0 + tile]
            yield crop, x0, y0


def iou(a: Det, b: Det) -> float:
    xA = max(a.x1, b.x1)
    yA = max(a.y1, b.y1)
    xB = min(a.x2, b.x2)
    yB = min(a.y2, b.y2)
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def nms(dets: List[Det], iou_thr: float = 0.5) -> List[Det]:
    """
    NMS simples por classe (remove duplicatas).
    """
    out: List[Det] = []
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)

    used = [False] * len(dets)
    for i, di in enumerate(dets):
        if used[i]:
            continue
        out.append(di)
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            dj = dets[j]
            if di.cls != dj.cls:
                continue
            if iou(di, dj) >= iou_thr:
                used[j] = True
    return out


def infer_tiled(
    model: YOLO,
    img_bgr: np.ndarray,
    tile: int = 640,
    overlap: float = 0.25,
    conf: float = 0.25,
    iou_thr_nms_global: float = 0.5,
) -> List[Det]:
    """
    Roda YOLO em tiles e junta as detecções em coords da imagem original.
    """
    all_dets: List[Det] = []

    for crop, x0, y0 in iter_tiles(img_bgr, tile=tile, overlap=overlap):
        # Ultralytics aceita numpy BGR direto
        r = model.predict(crop, conf=conf, verbose=False)[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
            # converte do tile pro global somando offset
            all_dets.append(
                Det(
                    cls=int(k),
                    conf=float(c),
                    x1=float(x1 + x0),
                    y1=float(y1 + y0),
                    x2=float(x2 + x0),
                    y2=float(y2 + y0),
                )
            )

    # NMS global pra remover duplicatas entre tiles
    return nms(all_dets, iou_thr=iou_thr_nms_global)


def draw_dets(img_bgr: np.ndarray, dets: List[Det], names: dict[int, str] | None = None) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
        label = f"{names.get(d.cls, d.cls) if names else d.cls}:{d.conf:.2f}"
        cv2.putText(out, label, (int(d.x1), max(0, int(d.y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    img_path = r"C:\_projects\dev\smart_inventory\out\vj120_2_jpg.rf.f21778973a710fe2a936d142170c8a57_annotated.jpg"
    model_path = "yolo11n.pt"  # ou seu best.pt treinado

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    model = YOLO(model_path)

    dets = infer_tiled(
        model=model,
        img_bgr=img,
        tile=640,
        overlap=0.25,
        conf=0.25,
        iou_thr_nms_global=0.5,
    )

    print(f"Detecções finais: {len(dets)}")
    annotated = draw_dets(img, dets, names=getattr(model, "names", None))
    cv2.imwrite("saida_annotated.jpg", annotated)
    print("Salvei: saida_annotated.jpg")
