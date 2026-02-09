from pathlib import Path
import yaml
 
# Caminho raiz do dataset
dataset_path = Path(r"C:\_projects\dev\smart_inventory\datasets\vaca_tilada")
 
# Classes do seu dataset
classes = [
    "cow"
]
 
# Estrutura do YAML
data = {
    "path": str(dataset_path),
    "train": r"images\train",
    "val": r"images\val",
    "nc": len(classes),
    "names": classes
}
 
# Escrita do arquivo YAML
yaml_path = dataset_path / "data.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False)
 
print(f"data.yaml gerado com sucesso em: {yaml_path}")