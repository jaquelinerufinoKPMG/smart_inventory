"""
Pipeline YOLO para contagem em dataset tilado (640x640 com overlap):
1) (Opcional) gera tiles + labels + tiles.csv com offsets
2) roda predict nos tiles
3) “des-tila”: converte bboxes do tile → coordenadas da imagem original usando offsets
4) deduplica por imagem original (NMS global por IoU)
5) salva:
   - counts.csv (contagem final por imagem original)
   - detections_merged.csv (bboxes finais por imagem original)
   - overlays/*.jpg (imagem original com caixas finais + COUNT)

Requisitos:
pip install ultralytics opencv-python pandas numpy

Obs:
- Este script assume que você tem as imagens originais disponíveis (base_in/images/<split>)
- Se você já gerou tiles, pode pular a etapa de tiling e só gerar o tiles.csv (ou usar o que foi gerado).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# Utils de bbox
# ============================================================

def xywhn_to_xyxy_px(xc, yc, w, h, img_w, img_h):
    """YOLO normalized (xc,yc,w,h) -> xyxy em pixels"""
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def clip_box(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    nx1 = max(x1, xmin)
    ny1 = max(y1, ymin)
    nx2 = min(x2, xmax)
    ny2 = min(y2, ymax)
    return nx1, ny1, nx2, ny2


def box_area_xyxy(b: np.ndarray) -> np.ndarray:
    """b: (N,4) xyxy"""
    w = np.maximum(0.0, b[:, 2] - b[:, 0])
    h = np.maximum(0.0, b[:, 3] - b[:, 1])
    return w * h


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IoU entre:
      a: (4,) xyxy
      b: (N,4) xyxy
    retorna (N,)
    """
    xA = np.maximum(a[0], b[:, 0])
    yA = np.maximum(a[1], b[:, 1])
    xB = np.minimum(a[2], b[:, 2])
    yB = np.minimum(a[3], b[:, 3])

    inter_w = np.maximum(0.0, xB - xA)
    inter_h = np.maximum(0.0, yB - yA)
    inter = inter_w * inter_h

    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = box_area_xyxy(b)

    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float = 0.55,
) -> List[int]:
    """
    NMS clássico. Retorna índices mantidos.
    boxes: (N,4) xyxy
    scores: (N,)
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = iou_xyxy(boxes[i], boxes[rest])
        rest = rest[ious <= iou_thr]
        order = rest

    return keep


# ============================================================
# Tiling (com tiles.csv)
# ============================================================

def iter_tiles_coords(w: int, h: int, tile: int, overlap: float):
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if xs[-1] != w - tile:
        xs.append(w - tile)
    if ys[-1] != h - tile:
        ys.append(h - tile)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, x0 + tile, y0 + tile


def tile_dataset_yolo_with_index(
    in_images_dir: Path,
    in_labels_dir: Path,
    out_dir: Path,
    split: str = "train",
    tile: int = 640,
    overlap: float = 0.25,
    min_box_area_px: float = 16.0,
    min_visibility: float = 0.30,
    save_empty_tiles: bool = True,
) -> Path:
    """
    Gera tiles + labels e também salva tiles.csv com offsets e metadados.
    Retorna o path do tiles.csv.
    """
    out_images = out_dir / "images" / split
    out_labels = out_dir / "labels" / split
    out_meta = out_dir / "tiles"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    tile_rows = []

    image_paths = sorted([p for p in in_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    for img_path in image_paths:
        label_path = in_labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] não li imagem: {img_path}")
            continue

        H, W = img.shape[:2]

        # Carrega boxes originais (se existirem)
        boxes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                bx1, by1, bx2, by2 = xywhn_to_xyxy_px(xc, yc, bw, bh, W, H)
                area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                boxes.append((cls, bx1, by1, bx2, by2, area))

        tile_idx = 0
        for x0, y0, x1t, y1t in iter_tiles_coords(W, H, tile=tile, overlap=overlap):
            crop = img[y0:y1t, x0:x1t]
            tile_labels: List[str] = []

            for cls, bx1, by1, bx2, by2, area_orig in boxes:
                cx1, cy1, cx2, cy2 = clip_box(bx1, by1, bx2, by2, x0, y0, x1t, y1t)
                inter_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
                if inter_area <= 0:
                    continue

                visibility = inter_area / max(area_orig, 1e-9)
                if visibility < min_visibility:
                    continue

                rx1 = cx1 - x0
                ry1 = cy1 - y0
                rx2 = cx2 - x0
                ry2 = cy2 - y0

                if (rx2 - rx1) * (ry2 - ry1) < min_box_area_px:
                    continue

                # xyxy px -> yolo normalized no tile
                bw = max(0.0, rx2 - rx1)
                bh = max(0.0, ry2 - ry1)
                xc = rx1 + bw / 2
                yc = ry1 + bh / 2
                tile_labels.append(f"{cls} {xc / tile:.6f} {yc / tile:.6f} {bw / tile:.6f} {bh / tile:.6f}")

            has_obj = len(tile_labels) > 0
            if (not has_obj) and (not save_empty_tiles):
                tile_idx += 1
                continue

            out_name = f"{img_path.stem}_t{tile_idx:04d}"
            out_img_path = out_images / f"{out_name}.jpg"
            out_lbl_path = out_labels / f"{out_name}.txt"

            cv2.imwrite(str(out_img_path), crop)
            out_lbl_path.write_text("\n".join(tile_labels) + ("\n" if tile_labels else ""), encoding="utf-8")

            tile_rows.append(
                dict(
                    tile_name=out_name,
                    tile_path=str(out_img_path),
                    orig_name=img_path.stem,
                    orig_path=str(img_path),
                    x0=int(x0), y0=int(y0), x1=int(x1t), y1=int(y1t),
                    orig_w=int(W), orig_h=int(H),
                    tile=int(tile),
                    overlap=float(overlap),
                )
            )

            tile_idx += 1

        print(f"[ok] {img_path.name}: {tile_idx} tiles iterados")

    tiles_csv = out_meta / f"{split}_tiles.csv"
    pd.DataFrame(tile_rows).to_csv(tiles_csv, index=False, encoding="utf-8")
    print(f"[ok] tiles.csv salvo em: {tiles_csv}")
    return tiles_csv


# ============================================================
# Predict + merge + count
# ============================================================

@dataclass
class PredictConfig:
    weights_path: Path
    tiles_csv: Path
    out_dir: Path
    split: str = "val"
    conf_thr: float = 0.35
    iou_thr_nms: float = 0.55
    class_id_cow: Optional[int] = 0  # se você só tem 1 classe, deixa 0
    imgsz: int = 640
    device: Optional[str] = None     # "cpu", "0", "mps" etc. Deixa None pro auto.
    max_det: int = 3000
    save_overlays: bool = True
    overlay_thickness: int = 2


def run_predict_merge_count(cfg: PredictConfig) -> Tuple[Path, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    # Lê tiles index
    tiles_df = pd.read_csv(cfg.tiles_csv)
    # sanity
    required_cols = {"tile_name","tile_path","orig_name","orig_path","x0","y0","orig_w","orig_h"}
    missing = required_cols - set(tiles_df.columns)
    if missing:
        raise ValueError(f"tiles.csv sem colunas necessárias: {missing}")

    model = YOLO(str(cfg.weights_path))

    # Agrupa tiles por imagem original
    grouped = tiles_df.groupby("orig_name", sort=False)

    merged_rows = []
    count_rows = []

    for orig_name, g in grouped:
        # Carrega imagem original (pra overlay)
        orig_path = Path(g["orig_path"].iloc[0])
        img_orig = cv2.imread(str(orig_path)) if cfg.save_overlays else None
        if cfg.save_overlays and img_orig is None:
            print(f"[warn] não consegui ler original p/ overlay: {orig_path}")

        all_boxes = []
        all_scores = []
        all_clss = []

        tile_paths = g["tile_path"].tolist()
        # Prediz em batch nos tiles dessa imagem original
        results = model.predict(
            source=tile_paths,
            imgsz=cfg.imgsz,
            conf=cfg.conf_thr,
            iou=0.7,                 # iou interno do YOLO (NMS por tile). Pode deixar 0.7.
            device=cfg.device,
            max_det=cfg.max_det,
            verbose=False,
        )

        # results está alinhado com tile_paths (mesma ordem)
        for row, r in zip(g.itertuples(index=False), results):
            x0, y0 = int(row.x0), int(row.y0)

            if r.boxes is None or len(r.boxes) == 0:
                continue

            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()          # (N,4) no tile
            conf = b.conf.cpu().numpy()          # (N,)
            cls_ = b.cls.cpu().numpy().astype(int)  # (N,)

            # Filtra classe (se aplicável)
            if cfg.class_id_cow is not None:
                mask = (cls_ == int(cfg.class_id_cow))
                xyxy = xyxy[mask]
                conf = conf[mask]
                cls_ = cls_[mask]

            if len(xyxy) == 0:
                continue

            # Offset -> coords na imagem original
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0

            # Clip na original (segurança)
            ow, oh = int(row.orig_w), int(row.orig_h)
            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, ow - 1)
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, ow - 1)
            xyxy[:, 1] = np.clip(xyxy[:, 1], 0, oh - 1)
            xyxy[:, 3] = np.clip(xyxy[:, 3], 0, oh - 1)

            all_boxes.append(xyxy)
            all_scores.append(conf)
            all_clss.append(cls_)

        if len(all_boxes) == 0:
            # Nenhuma detecção
            count_rows.append(dict(orig_name=orig_name, count=0, orig_path=str(orig_path)))
            if cfg.save_overlays and img_orig is not None:
                out_overlay = cfg.out_dir / "overlays" / f"{orig_name}_count0.jpg"
                # escreve COUNT 0
                cv2.putText(img_orig, "COUNT: 0", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                cv2.imwrite(str(out_overlay), img_orig)
            continue

        boxes = np.vstack(all_boxes).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)
        clss = np.concatenate(all_clss).astype(int)

        # NMS global na imagem original (dedupe)
        keep = nms_xyxy(boxes, scores, iou_thr=cfg.iou_thr_nms)
        boxes_k = boxes[keep]
        scores_k = scores[keep]
        clss_k = clss[keep]

        # (Opcional) remove boxes degeneradas
        area = box_area_xyxy(boxes_k)
        valid = area > 1.0
        boxes_k = boxes_k[valid]
        scores_k = scores_k[valid]
        clss_k = clss_k[valid]

        # Conta final
        count = int(len(boxes_k))
        count_rows.append(dict(orig_name=orig_name, count=count, orig_path=str(orig_path)))

        # Salva deteções finais
        for (x1, y1, x2, y2), sc, c in zip(boxes_k, scores_k, clss_k):
            merged_rows.append(
                dict(
                    orig_name=orig_name,
                    orig_path=str(orig_path),
                    cls=int(c),
                    conf=float(sc),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                )
            )

        # Overlay
        if cfg.save_overlays and img_orig is not None:
            for (x1, y1, x2, y2), sc in zip(boxes_k, scores_k):
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(img_orig, p1, p2, (0, 255, 0), cfg.overlay_thickness)
                cv2.putText(
                    img_orig,
                    f"{sc:.2f}",
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(img_orig, f"COUNT: {count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            out_overlay = cfg.out_dir / "overlays" / f"{orig_name}_count{count}.jpg"
            cv2.imwrite(str(out_overlay), img_orig)

        print(f"[ok] {orig_name}: {count} vacas (após NMS)")

    counts_csv = cfg.out_dir / f"{cfg.split}_counts.csv"
    merged_csv = cfg.out_dir / f"{cfg.split}_detections_merged.csv"

    pd.DataFrame(count_rows).to_csv(counts_csv, index=False, encoding="utf-8")
    pd.DataFrame(merged_rows).to_csv(merged_csv, index=False, encoding="utf-8")

    print(f"[ok] counts -> {counts_csv}")
    print(f"[ok] merged detections -> {merged_csv}")
    if cfg.save_overlays:
        print(f"[ok] overlays -> {cfg.out_dir / 'overlays'}")

    return counts_csv, merged_csv


# ============================================================
# MAIN (ajuste paths aqui)
# ============================================================

if __name__ == "__main__":
    # --------- Ajuste esses caminhos ----------
    base_in = Path(r"datasets\cristal")             # orig
    base_out_tiles = Path(r"datasets\cristal_cortada")  # tiles
    weights_path = Path(r"src\yolo models\train_91.pt")     # seu best.pt

    # --------- 1) (Opcional) gerar tiles + tiles.csv ----------
    # Se você já gerou os tiles, pode comentar isso e só apontar pro CSV (ou gerar só o CSV).
    tiles_csv_val = tile_dataset_yolo_with_index(
        in_images_dir=base_in / "images" / "val",
        in_labels_dir=base_in / "labels" / "val",
        out_dir=base_out_tiles,
        split="val",
        tile=640,
        overlap=0.25,
        min_box_area_px=16.0,
        min_visibility=0.30,
        save_empty_tiles=True,
    )

    # --------- 2) predict + merge + count ----------
    out_pred = base_out_tiles #Path(r"data\output\cow_count")

    cfg = PredictConfig(
        weights_path=weights_path,
        tiles_csv=tiles_csv_val,
        out_dir=out_pred,
        split="val",
        conf_thr=0.35,         # ajusta se tiver muito falso positivo
        iou_thr_nms=0.55,      # ajusta se estiver duplicando (↑) ou perdendo vaca (↓)
        class_id_cow=0,        # se sua classe vaca é 0
        imgsz=640,
        device=None,           # "cpu" / "0" / "mps"
        save_overlays=True,
    )

    run_predict_merge_count(cfg)
