# 再現手順チェックリスト

## 構成（簡略化後）

- `train.py` + `evaluate.py` のみ
- `hsi_dataset.py` のみ（random split 版は削除）
- Flower / Leaves 用 config: `flower.yaml`, `leaves.yaml`

## やるべきこと

### 1. データセット

- [ ] 前処理済み HDF5 (MatFlower60.h5, MatLeaves60.h5) を配置
- [ ] `compute_srgb_max.py` で srgb_max_values.json を生成

### 2. 学習

- [ ] `HFD100_DATA_DIR` 設定または config の `data_dir` 編集
- [ ] Flower: `python src/train.py --config src/configs/flower.yaml`
- [ ] Leaves: `python src/train.py --config src/configs/leaves.yaml`

### 3. 評価

- [ ] eval_config.yaml の `train_run_dir` を学習ディレクトリに設定
- [ ] `python src/evaluate.py --config ... --eval_config ...`
