import os
from dotenv import load_dotenv
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient

load_dotenv()

# ========= CONFIG =========
YOLO_FOLDER = os.getenv("YOLO_FOLDER", ".")  # base local
WEIGHTS_FOLDER = os.getenv("WEIGHTS_FOLDER", ".")  # base local
CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.getenv("STORAGE_CONTAINER_NAME", "meu-container")

# Prefixo no blob (pasta "virtual" onde está seu dataset)
BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "datasets/vaca_tilada/")

# Onde salvar localmente
LOCAL_DATASET_DIR = os.path.join("datasets", "vaca_tilada")

# Paths locais que o YOLO vai usar
DATA_YAML = os.path.abspath(os.path.join(LOCAL_DATASET_DIR, "data.yaml"))
WEIGHTS_PATH = os.path.abspath(os.path.join(WEIGHTS_FOLDER, "best.pt"))


def download_prefix_from_blob(conn_str, container_name, prefix, local_root):
    if not conn_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING não definido.")

    # garante prefix com barra no final
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    service = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(container_name)

    print(f"Baixando blobs de '{container_name}' com prefixo '{prefix}' para '{local_root}'...")

    os.makedirs(local_root, exist_ok=True)
    print("Diretório local criado:", os.path.abspath(local_root))

    for blob in container.list_blobs(name_starts_with=prefix):
        blob_name = blob.name
        print(f"Processando blob: {blob_name}")

        if blob_name.endswith("/"):
            continue

        # caminho relativo dentro do prefixo
        rel_path = blob_name[len(prefix):] if blob_name.startswith(prefix) else blob_name
        rel_path = rel_path.lstrip("/\\")  # evita virar path absoluto

        local_path = os.path.normpath(os.path.join(local_root, rel_path))

        # proteção extra: impede escapar do diretório base (segurança)
        base_abs = os.path.abspath(local_root) + os.sep
        local_abs = os.path.abspath(local_path)
        if not local_abs.startswith(base_abs):
            raise ValueError(f"Blob path suspeito (escapa do diretório base): {blob_name} -> {local_abs}")

        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(container.download_blob(blob_name).readall())

        print("  Baixado para:", local_path)

    return local_root

# ========= 1) Baixa dataset do Blob para local =========
download_prefix_from_blob(
    conn_str=CONN_STR,
    container_name=CONTAINER,
    prefix=BLOB_PREFIX,
    local_root=LOCAL_DATASET_DIR
)

# ========= 2) Treina normalmente =========
model = YOLO(WEIGHTS_PATH)

model.train(
    data=DATA_YAML,
    workers=0,
    epochs=60,
    imgsz=640,
    lr0=0.001,    # menor é mais seguro aqui
    patience=20,  # early stopping se parar de melhorar
)
