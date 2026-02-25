# Smart Inventory - Detecção de Vacas com YOLO

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange)
![Python](https://img.shields.io/badge/python-3.13-blue)
![YOLO](https://img.shields.io/badge/ultralytics-8.4.7-00ffff)
![Dataset](https://img.shields.io/badge/dataset-YOLO%20format-2ea44f)
![Licença](https://img.shields.io/badge/license-n%C3%A3o%20definida-lightgrey)

Pipeline de visão computacional para detecção de vacas em imagens, com foco em treino, avaliação e comparação de modelos YOLO.

## Sumário
- [Visão geral](#visão-geral)
- [Resultados atuais](#resultados-atuais)
- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Setup rápido](#setup-rápido)
- [Como usar](#como-usar)
- [Scripts auxiliares](#scripts-auxiliares)
- [Roadmap](#roadmap)
- [Limitações conhecidas](#limitações-conhecidas)

## Visão geral
Este projeto cobre o ciclo completo de detecção de objetos (classe `cow`) usando Ultralytics YOLO:
- ingestão de dataset a partir do Azure Blob Storage;
- preparação opcional com tiling de imagens;
- treino e retomada de treino a partir de checkpoints;
- análise de qualidade de imagem e comparação entre pesos treinados.

## Resultados atuais
Resumo dos melhores pontos de cada treino salvo em `runs/detect/*/results.csv` (critério: maior `mAP50-95(B)`).

| Run | Epoch (melhor) | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `train` | 109 | 0.73 | 0.46 | 0.57 | 0.26 |
| `train2` | 146 | 0.95 | 0.31 | 0.63 | 0.39 |
| `train3` | 60 | 0.86 | 0.81 | 0.86 | 0.63 |
| `train6` | 161 | 0.94 | 0.56 | 0.76 | 0.57 |

Referência atual de melhor equilíbrio geral: **`train3`**.

## Arquitetura do projeto
```text
smart_inventory/
|- train.py
|- train_tiled_images.py
|- restart_train.py
|- requirements.txt
|- datasets/
|- data/
|- runs/
`- src/
   |- azure.py
   |- create_tiled_images.py
   |- create_yaml.py
   |- generate_data_analysis.py
   |- extract_image_metrics.py
   |- build_profile.py
   `- draw_bounding_boxes.py
```

## Pré-requisitos
1. Python 3.13
2. Ambiente virtual
3. Dependências instaladas com:

```bash
pip install -r requirements.txt
```

## Setup rápido
1. Criar e ativar ambiente virtual.
2. Instalar dependências.
3. Criar arquivo `.env` na raiz:

```env
AZURE_STORAGE_CONNECTION_STRING="<sua_connection_string_do_azure_blob>"
```

## Como usar
### 1) Treino padrão
Baixa dataset do Azure (container `blobarcelor`, prefixo `vaca`) para `datasets/vaca` e inicia treino com `yolo26l.pt`.

```bash
python train.py
```

### 2) Treino com imagens tiladas
Executa treino usando `datasets/vaca_tilada`.

```bash
python train_tiled_images.py
```

### 3) Retomar treino
Retoma treino a partir de checkpoint salvo.

```bash
python restart_train.py
```

### 4) Retomar treino no Linux com script
Se o projeto estiver em `~/smart_inventory`, use o script auxiliar:

```bash
chmod +x smart_inventory/restart_train_linux.sh
smart_inventory/restart_train_linux.sh
```

O script:
- entra na pasta `~/smart_inventory`;
- ativa o ambiente virtual `.venv`;
- executa `python restart_train.py`.

## Scripts auxiliares
- `src/create_tiled_images.py`: gera tiles e recalcula labels YOLO por tile.
- `src/create_yaml.py`: gera `data.yaml` para dataset.
- `src/generate_data_analysis.py`: valida modelos e exporta análise em Excel.
- `src/extract_image_metrics.py`: extrai métricas de imagem (brilho, contraste, nitidez, entropia).
- `src/build_profile.py`: consolida estatísticas do dataset em JSON.
- `src/draw_bounding_boxes.py`: desenha boxes de labels YOLO em imagens.

## Roadmap
- [x] Pipeline de treino YOLO funcional
- [x] Download de dataset via Azure Blob
- [x] Geração de dataset tilado
- [x] Retomada de treino por checkpoint
- [x] Exportação de pesos (`.pt`, `.onnx`)
- [x] Gerar treino do yolo26l.pt para imagens inteiras
- [ ] Gerar treino do yolo26l.pt para imagens cortadas (Em andamento)
- [ ] Gerar treino do yolo26l.pt para imagens inteiras com data augmentation
- [ ] Gerar treino do yolo26l.pt para imagens cortadas com data augmentation
- [ ] Validação da qualidade de imagens 
      - Atenção à esse ponto já que precisa melhorar o dataset de treinamento/validação
         - As imagens do treinamento para essa validação devem ser o Ideal Perfeito (altura certa, ângulo certo)
      - O arquivo que contem esses testes é `image_validation.ipynb`
- [ ] Contagem precisa de imagens
      - Não está preciso
      - São os arquivos:
         - `train_tiled_images.py`
         - `test_count_tile_images.py`

## Limitações conhecidas
- Existem caminhos absolutos em alguns scripts (orientados ao ambiente Windows local).
- Há artefatos de treino e pesos grandes dentro do repositório.
- Não há licença definida no momento.
