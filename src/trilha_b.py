from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def yolo_to_xyxy(line: str, w: int, h: int) -> Tuple[int, float, float, float, float]:
    """
    Converte linha YOLO (normalizada) -> xyxy em pixels.
    Retorna (cls, x1,y1,x2,y2)
    """
    parts = line.strip().split()
    cls = int(parts[0])
    xc, yc, bw, bh = map(float, parts[1:5])

    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    """
    Converte xyxy em pixels -> YOLO normalizado no frame w×h
    """
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return f"{cls} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def clip_box(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    nx1 = max(x1, xmin)
    ny1 = max(y1, ymin)
    nx2 = min(x2, xmax)
    ny2 = min(y2, ymax)
    return nx1, ny1, nx2, ny2


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


def tile_dataset_yolo(
    in_images_dir: Path,
    in_labels_dir: Path,
    out_dir: Path,
    split: str = "train",
    tile: int = 640,
    overlap: float = 0.25,
    min_box_area_px: float = 16.0,   # filtra box ridícula
    min_visibility: float = 0.30,    # % mínima da área original que precisa aparecer no tile
):
    out_images = out_dir / "images" / split
    out_labels = out_dir / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in in_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    for img_path in image_paths:
        label_path = in_labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] não li imagem: {img_path}")
            continue

        H, W = img.shape[:2]

        boxes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls, x1, y1, x2, y2 = yolo_to_xyxy(line, W, H)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                boxes.append((cls, x1, y1, x2, y2, area))
        else:
            # imagem sem label: ok (gera tiles sem objetos)
            boxes = []

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

                # coords relativas ao tile
                rx1 = cx1 - x0
                ry1 = cy1 - y0
                rx2 = cx2 - x0
                ry2 = cy2 - y0

                if (rx2 - rx1) * (ry2 - ry1) < min_box_area_px:
                    continue

                tile_labels.append(xyxy_to_yolo(cls, rx1, ry1, rx2, ry2, tile, tile))

            # salva tile (mesmo sem label, dependendo do teu caso você pode querer salvar ou pular)
            out_name = f"{img_path.stem}_t{tile_idx:04d}"
            out_img_path = out_images / f"{out_name}.jpg"
            out_lbl_path = out_labels / f"{out_name}.txt"

            cv2.imwrite(str(out_img_path), crop)

            # se não tiver objetos, salva vazio (YOLO aceita) ou não salva o txt
            out_lbl_path.write_text("\n".join(tile_labels) + ("\n" if tile_labels else ""), encoding="utf-8")

            tile_idx += 1

        print(f"[ok] {img_path.name}: {tile_idx} tiles gerados")


if __name__ == "__main__":
    # Exemplo de uso:
    # dataset_original/
    #   images/train/*.jpg
    #   labels/train/*.txt
    base_in = Path(r"C:\_projects\dev\smart_inventory\datasets\vaca")
    base_out = Path(r"C:\_projects\dev\smart_inventory\datasets\vaca_tilada")

    tile_dataset_yolo(
        in_images_dir=base_in / "images" / "train",
        in_labels_dir=base_in / "labels" / "train",
        out_dir=base_out,
        split="train",
        tile=640,
        overlap=0.25,
        min_box_area_px=16.0,
        min_visibility=0.30,
    )

    tile_dataset_yolo(
         in_images_dir=base_in / "images" / "val",
         in_labels_dir=base_in / "labels" / "val",
         out_dir=base_out,
         split="val",
         tile=640,
         overlap=0.25,
         min_box_area_px=16.0,
         min_visibility=0.30,
     )

    print("Pronto: vaca_tilada/")
