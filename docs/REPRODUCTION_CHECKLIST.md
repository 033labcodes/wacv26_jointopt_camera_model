# Reproduction Checklist

## Overview

- `train.py` + `evaluate.py` only
- Single dataset loader: `hsi_dataset.py`
- Flower / Leaves configs: `flower.yaml`, `leaves.yaml`

## Tasks

### 1. Dataset and Preprocessing

- [ ] Download original HFD100 dataset
- [ ] Convert `.mat` to HDF5 (`MatFlower60.h5`, `MatLeaves60.h5`)
- [ ] Run `compute_srgb_max.py` to generate sRGB max JSON files
- [ ] Set `HFD100_DATA_DIR` or `data_dir` in config to the preprocessed data path

### 2. Training

- [ ] Flower: `python src/train.py --config src/configs/flower.yaml`
- [ ] Leaves: `python src/train.py --config src/configs/leaves.yaml`

### 3. Evaluation

- [ ] Set `train_run_dir` in `src/configs/eval_config.yaml` to the checkpoint directory
- [ ] Run: `python src/evaluate.py --config src/configs/flower.yaml --eval_config src/configs/eval_config.yaml`
