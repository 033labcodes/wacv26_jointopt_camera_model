#!/bin/bash
# 評価実行例
# eval_config.yaml の train_run_dir を学習済みチェックポイントのパスに編集してから実行

set -e
cd "$(dirname "$0")/.."

# Flower: --config src/configs/flower.yaml
# Leaves: --config src/configs/leaves.yaml

python src/evaluate.py \
  --config src/configs/flower.yaml \
  --eval_config src/configs/eval_config.yaml
