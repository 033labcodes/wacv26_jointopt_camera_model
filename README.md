# Joint Optimization of Camera Model and Deep Neural Network for Image Recognition (WACV 2026)

**HFD100 ハイパースペクトル画像分類**の再現用リポジトリです．

## 構成

- `train` + `evaluate` のみ
- Flower / Leaves に対応
- 前処理済み HDF5 を前提（.mat 変換はスキップ）

## セットアップ

```bash
poetry install
```

詳細は [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) を参照．

## 実行

### 学習

```bash
# Flower
python src/train.py --config src/configs/flower.yaml

# Leaves
python src/train.py --config src/configs/leaves.yaml
```

または `./scripts/run_train.sh`（中身を flower/leaves で切り替え可能）

### 評価

`src/configs/eval_config.yaml` の `train_run_dir` を学習済みチェックポイントのパスに設定してから：

```bash
python src/evaluate.py --config src/configs/flower.yaml --eval_config src/configs/eval_config.yaml
```

## 主要ファイル

| ファイル | 用途 |
|----------|------|
| `src/train.py` | 学習 |
| `src/evaluate.py` | 評価 |
| `src/hsi_dataset.py` | データセット |
| `src/configs/flower.yaml` | Flower 用設定 |
| `src/configs/leaves.yaml` | Leaves 用設定 |

## 環境変数

| 変数 | 説明 |
|------|------|
| `HFD100_DATA_DIR` | HDF5 (Mat*.h5) のディレクトリ |
