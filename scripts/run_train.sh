#!/bin/bash
# 学習実行例
# 使用前に HFD100_DATA_DIR を設定するか，config の data_dir を編集してください

set -e
cd "$(dirname "$0")/.."

# 例: export HFD100_DATA_DIR=/path/to/HFD100_Mat_dataset

# Flower: python src/train.py --config src/configs/flower.yaml
# Leaves: python src/train.py --config src/configs/leaves.yaml

python src/train.py --config src/configs/flower.yaml
