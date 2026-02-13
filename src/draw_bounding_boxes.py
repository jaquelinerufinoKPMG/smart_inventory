from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass
class Det:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def read_yolo_label_txt(txt_path: Path, img_w: int, img_h: int, conf: float = 1.0) -> list[Det]:
    """
    Lê arquivo .txt no formato YOLO:
    cls x_center y_center width height  (tudo normalizado 0..1)
    e converte para xyxy em pixels.
    """
    dets: list[Det] = []
    if not txt_path.exists():
        return dets

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        # clamp pra não sair pra fora da imagem
        x1 = max(0.0, min(x1, img_w - 1))
        y1 = max(0.0, min(y1, img_h - 1))
        x2 = max(0.0, min(x2, img_w - 1))
        y2 = max(0.0, min(y2, img_h - 1))

        dets.append(Det(cls=cls, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))

    return dets


def draw_boxes(img, dets: list[Det], show_label: bool = False):
    out = img.copy()
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
        if show_label:
            cv2.putText(
                out,
                f"{d.cls}",
                (int(d.x1), max(0, int(d.y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return out


def main():
    images_dir = Path(r"datasets\teste")
    out_dir = Path(r"datasets\out")
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] não consegui ler {img_path.name}")
            continue

        h, w = img.shape[:2]

        # procura o txt com o mesmo nome da imagem
        txt_path = img_path.with_suffix(".txt")
        dets = read_yolo_label_txt(txt_path, img_w=w, img_h=h, conf=1.0)

        if not dets:
            print(f"[warn] sem dets (txt não existe ou vazio): {img_path.name}")
            continue

        annotated = draw_boxes(img, dets, show_label=False)  # <-- True pra mostrar classe
        out_path = out_dir / f"{img_path.stem}_boxes.jpg"
        cv2.imwrite(str(out_path), annotated)

        print(f"[ok] {img_path.name} + {txt_path.name} -> {len(dets)} boxes -> {out_path.name}")


if __name__ == "__main__":
    main()


