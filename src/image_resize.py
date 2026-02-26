import cv2
from pathlib import Path

def resize_image(input_path: str, output_path: str, width: int, height: int):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Lê a imagem
    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError("Imagem não encontrada ou inválida.")

    # Redimensiona
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Cria pasta se não existir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Salva imagem
    cv2.imwrite(str(output_path), resized)

    print(f"Imagem salva em: {output_path}")
    print(f"Nova resolução: {width}x{height}")


# Exemplo de uso

image_folder = Path(r"datasets\vaca\images\train")
output_image_folder = Path(r"datasets\vaca_pequena")

for img in image_folder.rglob("*"):
    if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        output_path = Path.joinpath(output_image_folder, img.name)
        resize_image(
            input_path=img,
            output_path=output_path,
            width=640,
            height=640
        )
        