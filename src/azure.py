import os

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()


CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

class downloadData:

    def __init__(self, container_name):
        service = BlobServiceClient.from_connection_string(CONN_STR)

        self.container = service.get_container_client(container_name)


    def download_files(self, prefix:str, local_root:str, skip_if_exists=True, verify_size=True):
        # garante prefix com barra no final
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        print(f"Baixando arquivos de '{self.container}' com prefixo '{prefix}' para '{local_root}'...")

        os.makedirs(local_root, exist_ok=True)
        print("Diretório local criado:", os.path.abspath(local_root))

        baixados = 0
        pulados = 0

        for blob in self.container.list_blobs(name_starts_with=prefix):
            blob_name = blob.name

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

            # se já existe, pode pular
            if skip_if_exists and os.path.exists(local_path):
                if verify_size:
                    try:
                        local_size = os.path.getsize(local_path)
                    except OSError:
                        local_size = -1

                    remote_size = getattr(blob, "size", None)  # list_blobs costuma trazer
                    if remote_size is not None and local_size == remote_size:
                        pulados += 1
                        print(f"Pulando (já existe e tamanho bate): {local_path}")
                        continue
                else:
                    pulados += 1
                    continue

            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            print(f"Baixando: {blob_name} -> {local_path}")
            with open(local_path, "wb") as f:
                f.write(self.container.download_blob(blob_name).readall())

            baixados += 1

        print(f"Download finalizado. Baixados: {baixados} | Pulados: {pulados}")
        return local_root