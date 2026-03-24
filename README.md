# Joint Optimization of Camera Model and Deep Neural Network for Image Recognition (WACV 2026)

Official implementation for HFD100 hyperspectral image classification with joint optimization of camera model and a classification network.

**Paper:** [Joint Optimization of Camera Model and Deep Neural Network for Image Recognition](https://openaccess.thecvf.com/content/WACV2026/papers/Noboru_Joint_Optimization_of_Camera_Model_and_Deep_Neural_Network_for_WACV_2026_paper.pdf) (WACV 2026)


## Overview

- Train and evaluate only (`train.py` + `evaluate.py`)
- Supports HFD100 Flower and Leaves datasets
- **Preprocessing required:** Download the original dataset, convert to HDF5, then compute sRGB max values (see below)

## Setup

```bash
poetry install
```

## Dataset and Preprocessing

**Preprocessing must be run first** before training and evaluation.

### 1. Download Original Dataset

Download the [HFD100 dataset](https://github.com/ying-fu/HFD100) (Zheng, Zhang & Fu, *Knowledge-Based Systems*, 2022). The dataset (~55GB, .mat format) is hosted on [Baidu Cloud](https://pan.baidu.com/s/1XKGafuIbFpI3V77LSeFbOA).

### 2. Convert to HDF5

Convert `.mat` files to HDF5 format to produce:

- `MatFlower60.h5`
- `MatLeaves60.h5`
- (Optional) `MatScenes60.h5`

Use the provided conversion script:

```bash
python data/hfd100tohdf5.py \
  --input-root /path/to/HFD100_Mat_dataset \
  --output-dir ./data \
  --datasets flower leaves
```

The generated HDF5 files have the following structure:

```
train/
  metadata      (JSON)
  hs/
    image0001, image0002, ...
test/
  metadata
  hs/
    ...
```

### 3. Compute sRGB Max Values

```bash
export HFD100_DATA_DIR=/path/to/your/hdf5/directory

python data/compute_srgb_max.py HFD100_Flower -d $HFD100_DATA_DIR -o data/HFD100_Flower_srgb_max_values.json
python data/compute_srgb_max.py HFD100_Leaves -d $HFD100_DATA_DIR -o data/HFD100_Leaves_srgb_max_values.json
```

Output JSON files can be placed in either `$HFD100_DATA_DIR` or `data/` in the project root.

## Quick Start

### Training

```bash
export HFD100_DATA_DIR=/path/to/your/hdf5/directory

# Flower
python src/train.py --config src/configs/flower.yaml

# Leaves
python src/train.py --config src/configs/leaves.yaml
```

### Evaluation

1. Edit `src/configs/eval_config.yaml`:

```yaml
run_parameters:
  train_run_dir: runs/train/your_run_name
  checkpoint: best_model  # or latest
  output_dir: flower_eval
```

2. Run:

```bash
python src/evaluate.py --config src/configs/flower.yaml --eval_config src/configs/eval_config.yaml
```

## Pretrained Weights

Classification backbones (ResNet-18, ViT-S/16, SE-ResNet50) pretrained on HFD100 Flower/Leaves are hosted on [Hugging Face](https://huggingface.co/dekkaiinu/wacv26_jointopt_pretrained) and downloaded automatically based on `classification_model` and `dataset_name`.

## Main Files

| File | Purpose |
|------|---------|
| `src/train.py` | Training |
| `src/evaluate.py` | Evaluation |
| `src/hsi_dataset.py` | HFD100 dataset loader |
| `data/compute_srgb_max.py` | Preprocessing: sRGB max value computation |
| `src/configs/flower.yaml` | Flower config |
| `src/configs/leaves.yaml` | Leaves config |

## Repository Layout

```
src/
├── train.py
├── evaluate.py
├── hsi_dataset.py
├── configs/
│   ├── flower.yaml
│   ├── leaves.yaml
│   └── eval_config.yaml
├── models/
├── utils/
└── camera_parameters/
data/
├── compute_srgb_max.py
├── HFD100_Flower_srgb_max_values.json  (generated)
└── HFD100_Leaves_srgb_max_values.json  (generated)
```

## Config / Environment

| Config / Arg | Description |
|--------------|-------------|
| `data_dir` | HDF5 directory. Config or `--data_dir`. Default: `./data` (or `HFD100_DATA_DIR`) |
| `srgb_max_dir` | Directory for sRGB max JSON files. Config or `--srgb_max_dir`. Default: same as `data_dir` |

## Citation

```bibtex
@InProceedings{Noboru_2026_WACV,
    author    = {Noboru, Youta and Ozasa, Yuko and Tanaka, Masayuki},
    title     = {Joint Optimization of Camera Model and Deep Neural Network for Image Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7626-7635}
}
```
