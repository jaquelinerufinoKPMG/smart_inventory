import os
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient

# ========= CONFIG =========
YOLO_FOLDER = os.getenv("YOLO_FOLDER", ".")  # base local
CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "meu-container")

# Prefixo no blob (pasta "virtual" onde está seu dataset)
BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "datasets/vaca_tilada/")

# Onde salvar localmente
LOCAL_DATASET_DIR = os.path.join(YOLO_FOLDER, "datasets", "vaca_tilada")

# Paths locais que o YOLO vai usar
DATA_YAML = os.path.abspath(os.path.join(LOCAL_DATASET_DIR, "data.yaml"))
WEIGHTS_PATH = os.path.abspath(os.path.join(YOLO_FOLDER, "yolo26n.pt"))


def download_prefix_from_blob(conn_str, container_name, prefix, local_root):
    """
    Baixa todos os blobs cujo nome começa com `prefix` para dentro de `local_root`,
    preservando subpastas.
    """
    if not conn_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING não definido.")

    service = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(container_name)

    # garante pasta base
    os.makedirs(local_root, exist_ok=True)

    for blob in container.list_blobs(name_starts_with=prefix):
        blob_name = blob.name

        # ignora "pastas" vazias se existirem
        if blob_name.endswith("/"):
            continue

        # Ex: prefix="datasets/vaca_tilada/" e blob="datasets/vaca_tilada/images/train/abc.jpg"
        # rel="images/train/abc.jpg"
        rel_path = blob_name[len(prefix):] if blob_name.startswith(prefix) else blob_name

        local_path = os.path.join(local_root, rel_path)
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)

        # baixa arquivo
        with open(local_path, "wb") as f:
            data = container.download_blob(blob_name).readall()
            f.write(data)

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
    epochs=120,
    imgsz=512,
    batch=1,
    workers=0,
    cache=False
)
