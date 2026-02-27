#!/bin/bash
# Evaluation script
# Edit train_run_dir in eval_config.yaml to point to the checkpoint directory

set -e
cd "$(dirname "$0")/.."

# Flower: --config src/configs/flower.yaml
# Leaves: --config src/configs/leaves.yaml

python src/evaluate.py \
  --config src/configs/flower.yaml \
  --eval_config src/configs/eval_config.yaml
