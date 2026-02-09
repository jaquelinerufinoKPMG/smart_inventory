import os
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient

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
    """
    Baixa todos os blobs cujo nome começa com `prefix` para dentro de `local_root`,
    preservando subpastas.
    """
    if not conn_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING não definido.")

    service = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(container_name)
    print(f"Baixando blobs de '{container_name}' com prefixo '{prefix}' para '{local_root}'...")

    # garante pasta base
    os.makedirs(local_root, exist_ok=True)
    print(container)
    for blob in container.list_blobs():
        print(f"Processando blob: {blob}")


# ========= 1) Baixa dataset do Blob para local =========
download_prefix_from_blob(
    conn_str=CONN_STR,
    container_name=CONTAINER,
    prefix=BLOB_PREFIX,
    local_root=LOCAL_DATASET_DIR
)