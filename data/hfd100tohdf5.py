import argparse
import h5py
import json
import os
from typing import List, Tuple

import numpy as np
import scipy.io
from tqdm import tqdm


def add_files_to_group(group: h5py.Group, hs_files: List[str], targets: List[int]) -> None:
    """Add HSI data and targets to an HDF5 group in the format expected by this project."""
    metadata = []
    for i, (hs_file, target) in tqdm(
        enumerate(zip(hs_files, targets)), total=len(hs_files), desc=f"Writing {group.name}"
    ):
        try:
            mat_dict = scipy.io.loadmat(hs_file)
        except Exception as e:
            print(f"Error loading MAT file {hs_file}: {e}")
            continue

        if "truth" not in mat_dict:
            print(f"Warning: 'truth' key not found in {hs_file}. Skipping.")
            continue

        data = mat_dict["truth"]
        group.create_dataset(
            f"hs/image{i}",
            data=data,
            compression="gzip",
            compression_opts=4,
        )
        group.create_dataset(f"target/image{i}", data=int(target))
        metadata.append({"hsi": f"hs/image{i}", "target": int(target)})

    group.create_dataset("metadata", data=json.dumps(metadata))


def collect_hfd100_paths(dataset_root: str) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Collect HFD100 MAT file paths and class indices from the original folder structure.

    Expected structure under dataset_root:
      dataset_root/
        Train/
          class_0/*.mat
          class_1/*.mat
          ...
        Test/
          class_0/*.mat
          class_1/*.mat
          ...
    """
    sets = ["Train", "Test"]
    train_img_path: List[str] = []
    train_targets: List[int] = []
    test_img_path: List[str] = []
    test_targets: List[int] = []

    for split in sets:
        img_path: List[str] = []
        targets: List[int] = []
        load_dir = os.path.join(dataset_root, split)
        class_dirs = [d for d in os.listdir(load_dir) if not d.startswith("._")]
        class_dirs = sorted(class_dirs)

        for class_index, cls in enumerate(class_dirs):
            class_dir = os.path.join(load_dir, cls)
            files = [f for f in os.listdir(class_dir) if not f.startswith("._")]
            for file in files:
                if file.endswith(".mat"):
                    img_path.append(os.path.join(class_dir, file))
                    targets.append(class_index)

        if split == "Train":
            train_img_path = img_path
            train_targets = targets
        elif split == "Test":
            test_img_path = img_path
            test_targets = targets

    # In this project we use train/val/test split derived from Train/Test,
    # but here we just keep train and test; validation will be derived later.
    valid_img_path = test_img_path
    valid_targets = test_targets
    return train_img_path, train_targets, valid_img_path, valid_targets, test_img_path, test_targets


DATASET_FOLDERS = {
    "flower": "MatFlower60",
    "leaves": "MatLeaves60",
    "scenes": "MatScenes60",
}


def process_dataset(name: str, input_root: str, output_dir: str) -> None:
    folder = DATASET_FOLDERS[name]
    dataset_root = os.path.join(input_root, folder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{folder}.h5")

    print(f"Processing {name} ({folder})")
    print(f"  Input MAT root: {dataset_root}")
    print(f"  Output HDF5:    {output_path}")

    train_img_path, train_targets, _, _, test_img_path, test_targets = collect_hfd100_paths(dataset_root)

    with h5py.File(output_path, "w") as hdf5_file:
        train_group = hdf5_file.create_group("train")
        add_files_to_group(train_group, train_img_path, train_targets)

        test_group = hdf5_file.create_group("test")
        add_files_to_group(test_group, test_img_path, test_targets)

    print(f"Finished {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HFD100 MAT dataset to HDF5 format used by this repository."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing MatFlower60 / MatLeaves60 / MatScenes60 folders from HFD100.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to save Mat*.h5 files (default: ./data).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["flower", "leaves"],
        choices=["flower", "leaves", "scenes", "all"],
        help='Datasets to convert (choices: "flower", "leaves", "scenes", "all"). Default: flower leaves',
    )
    args = parser.parse_args()

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["flower", "leaves", "scenes"]

    for name in datasets:
        process_dataset(name, args.input_root, args.output_dir)


if __name__ == "__main__":
    main()

import h5py
import scipy.io
import numpy as np
import json
import os
from tqdm import tqdm
from PIL import Image


def add_files_to_group(group, hs_files, target):
    metadata = []
    # hsファイルを追加
    for i, (hs_file, target) in tqdm(enumerate(zip(hs_files, target)), total=len(hs_files)):
        try:
            dict = scipy.io.loadmat(hs_file)
            # ここに画像を保存する処理を追加
        except Exception as e:
            print(f"Error loading MAT file {hs_file}: {e}")
            continue
        data = dict['truth']
        group.create_dataset(f'hs/image{i}', data=data, compression='gzip', compression_opts=4)
        group.create_dataset(f'target/image{i}', data=target)
        metadata.append({'hsi': f'hs/image{i}', 'target': target})
    group.create_dataset('metadata', data=json.dumps(metadata))


def split_hfd100(dir_name, dataset_path):
    sets = ['Train', 'Test']
    for set in sets:
        img_path = []
        target = []
        load_dir = os.path.join(dataset_path, dir_name, set)
        cls_dir = [d for d in os.listdir(load_dir) if not d.startswith('._')]
        cls_dir = sorted(cls_dir)
        for i, cls in enumerate(cls_dir):
            files = os.listdir(os.path.join(load_dir, cls))
            for file in files:
                if not file.startswith('._'):
                    img_path.append(os.path.join(load_dir, cls, file))
                    target.append(i)
        if set == 'Train':
            train_img_path = img_path
            train_targets = target
        elif set == 'Test':
            test_img_path = img_path
            test_targets = target
    valid_img_path = test_img_path
    valid_targets = test_targets
    return train_img_path, train_targets, valid_img_path, valid_targets, test_img_path, test_targets


with h5py.File('/mnt/hdd_vcml/HFD100_Mat_dataset/MatFlower60.h5', 'w') as hdf5_file:
    train_img_path, train_targets, _, _, test_img_path, test_targets = split_hfd100('MatFlower60', dataset_path='/mnt/hdd_vcml/HFD100_Mat_dataset/HFD100_Mat_dataset')
    print('MatFlower60 processing...')
    group = hdf5_file.create_group('train')
    metadata = add_files_to_group(group, train_img_path, train_targets)
    group = hdf5_file.create_group('test')
    metadata = add_files_to_group(group, test_img_path, test_targets)

with h5py.File('/mnt/hdd_vcml/HFD100_Mat_dataset/MatLeaves60.h5', 'w') as hdf5_file:
    train_img_path, train_targets, _, _, test_img_path, test_targets = split_hfd100('MatLeaves60', dataset_path='/mnt/hdd_vcml/HFD100_Mat_dataset/HFD100_Mat_dataset')
    print('MatLeaves60 processing...')
    group = hdf5_file.create_group('train')
    add_files_to_group(group, train_img_path, train_targets)
    group = hdf5_file.create_group('test')
    add_files_to_group(group, test_img_path, test_targets)

with h5py.File('/mnt/hdd_vcml/HFD100_Mat_dataset/MatScenes60.h5', 'w') as hdf5_file:
    train_img_path, train_targets, _, _, test_img_path, test_targets = split_hfd100('MatScenes60', dataset_path='/mnt/hdd_vcml/HFD100_Mat_dataset/HFD100_Mat_dataset')
    print('MatScenes60 processing...')
    group = hdf5_file.create_group('train')
    add_files_to_group(group, train_img_path, train_targets)
    group = hdf5_file.create_group('test')
    add_files_to_group(group, test_img_path, test_targets)
