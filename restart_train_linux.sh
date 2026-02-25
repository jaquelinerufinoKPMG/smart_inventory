#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/smart_inventory"

cd "$PROJECT_DIR"
source ".venv/bin/activate"
python "restart_train.py"
