# HFD100 分類 セットアップガイド（WACV 2026）

HFD100 ハイパースペクトル画像分類の再現手順です．  
**前処理済み HDF5 を前提**とし，.mat からの変換はスキップします．

---

## 1. データセット

### 1.1 前処理済み HDF5

以下のファイルが `data_dir` に配置されている想定です．

- `MatFlower60.h5`
- `MatLeaves60.h5`

**HDF5 構造:**
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

### 1.2 sRGB max values

```bash
python data/compute_srgb_max.py HFD100_Flower -d /path/to/data_dir -o data/HFD100_Flower_srgb_max_values.json
python data/compute_srgb_max.py HFD100_Leaves -d /path/to/data_dir -o data/HFD100_Leaves_srgb_max_values.json
```

---

## 2. 学習

```bash
export HFD100_DATA_DIR=/path/to/HFD100_Mat_dataset  # または config の data_dir を編集

# Flower
python src/train.py --config src/configs/flower.yaml

# Leaves
python src/train.py --config src/configs/leaves.yaml
```

---

## 3. 評価

`src/configs/eval_config.yaml` を編集：

```yaml
run_parameters:
  train_run_dir: runs/train/実際の学習ディレクトリ名
  output_dir: flower_eval
  ...
```

実行：

```bash
python src/evaluate.py --config src/configs/flower.yaml --eval_config src/configs/eval_config.yaml
```

---

## 4. リポジトリ構成

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
```
