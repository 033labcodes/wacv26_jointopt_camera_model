#!/bin/bash
# HFD100 データセットのダウンロード案内

echo ""
echo "=== HFD100 Dataset Download ==="
echo ""
echo "URL: https://pan.baidu.com/s/1XKGafuIbFpI3V77LSeFbOA"
echo "Access code: epou"
echo ""
echo "1. Download .mat files from Baidu Cloud"
echo "2. Convert .mat to HDF5 (MatFlower60.h5, MatLeaves60.h5, MatScenes60.h5)"
echo "3. Set HFD100_DATA_DIR or config data_dir to the HDF5 directory"
echo "4. Run: python data/compute_srgb_max.py HFD100_Flower -o data/HFD100_Flower_srgb_max_values.json"
echo ""
