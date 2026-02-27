#!/bin/bash
# Training script
# Set HFD100_DATA_DIR or edit data_dir in config before running

set -e
cd "$(dirname "$0")/.."

# Example: export HFD100_DATA_DIR=/path/to/preprocessed/hdf5/directory

# Flower: python src/train.py --config src/configs/flower.yaml
# Leaves: python src/train.py --config src/configs/leaves.yaml

python src/train.py --config src/configs/flower.yaml
