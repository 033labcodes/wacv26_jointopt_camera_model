import random
import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from utils.slice_wavelength import slice_hsi_spectrum
from tqdm import tqdm
import pickle
import hashlib


class HFD100_Dataset(Dataset):
    def __init__(self, dataset_name, dataset_type:str, camera_name:str, transform=None, val_ratio=0.1, seed=42, data_dir=None, srgb_max_dir=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir or os.environ.get('HFD100_DATA_DIR', './data')
        self.srgb_max_dir = srgb_max_dir
        self.dataset_type = dataset_type
        self.camera_name = camera_name
        self.transform = transform
        self.val_ratio = val_ratio
        self.seed = seed
        self.hdf5_path = self._get_hdf5_path()
        
        # Cache for loaded HSI data
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.hsi_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache key: dataset_name, dataset_type, val_ratio, seed (transform/camera_name don't affect loaded data)
        cache_key_parts = [
            str(self.dataset_name),
            str(self.dataset_type),
            str(self.val_ratio),
            str(self.seed)
        ]
        cache_filename = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest() + ".pkl"
        self.cache_file_path = os.path.join(cache_dir, cache_filename)
        
        # Load metadata first to determine which HSI paths are needed
        temp_h5_file = h5py.File(self.hdf5_path, 'r')
        if self.dataset_type == 'test':
            self.hsi_prefix = 'test'
            # Load all metadata for the specified dataset type
            self.meta_data_all_for_type = json.loads(temp_h5_file[f'{self.hsi_prefix}/metadata'][()])
        else: # train or val
            self.hsi_prefix = 'train' # train and val come from the 'train' group in HDF5
            self.meta_data_all_for_type = json.loads(temp_h5_file[f'{self.hsi_prefix}/metadata'][()])
        
        # Prepare full list of targets and HSI paths corresponding to meta_data_all_for_type
        targets_all_for_type = [data['target'] for data in self.meta_data_all_for_type]
        # Store relative paths for keys, full paths will be used for loading from HDF5
        # The keys for loaded_hsi_data will be the full HDF5 internal path (e.g., "train/hs/image0001")
        hsi_internal_paths_all_for_type = [f"{self.hsi_prefix}/{data['hsi']}" for data in self.meta_data_all_for_type]

        # Handle train/val split if applicable
        if self.dataset_type in ['train', 'val']:
            num_total_samples = len(self.meta_data_all_for_type)
            indices = list(range(num_total_samples))
            random.seed(seed)
            random.shuffle(indices)
            val_split_size = int(num_total_samples * val_ratio)
            
            current_selection_indices = []
            if self.dataset_type == 'train':
                current_selection_indices = indices[val_split_size:]
            else:  # val
                current_selection_indices = indices[:val_split_size]
            
            self.meta_data = [self.meta_data_all_for_type[i] for i in current_selection_indices]
            self.targets = [targets_all_for_type[i] for i in current_selection_indices]
            # These are the HDF5 internal paths for the current dataset split (train or val)
            self._hsi_keys_to_load_in_memory = [hsi_internal_paths_all_for_type[i] for i in current_selection_indices]
        else:  # test set
            self.meta_data = self.meta_data_all_for_type
            self.targets = targets_all_for_type
            self._hsi_keys_to_load_in_memory = hsi_internal_paths_all_for_type

        # Load from cache or HDF5
        if os.path.exists(self.cache_file_path):
            print(f"Loading HSI data from cache: {self.cache_file_path}")
            try:
                with open(self.cache_file_path, 'rb') as f:
                    self.loaded_hsi_data = pickle.load(f)
                print("HSI data loading from cache complete.")
            except Exception as e:
                print(f"Error loading from cache (will reload from HDF5): {e}")
                self._load_hsi_data_from_hdf5(temp_h5_file)
        else:
            print(f"Cache not found. Loading HSI data from HDF5: {self.hdf5_path}")
            self._load_hsi_data_from_hdf5(temp_h5_file)
        
        temp_h5_file.close() # Close HDF5 file after all data is loaded or if only metadata was needed and cache hit

        # Load sRGB max values: srgb_max_dir > data_dir > project data/
        candidates = []
        if self.srgb_max_dir:
            candidates.append(os.path.abspath(self.srgb_max_dir))
        candidates.append(os.path.abspath(self.data_dir))
        candidates.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
        srgb_max_json_path = None
        for d in candidates:
            p = os.path.join(d, f"{self.dataset_name}_srgb_max_values.json")
            if os.path.exists(p):
                srgb_max_json_path = p
                break
        if srgb_max_json_path is None:
            srgb_max_json_path = os.path.join(candidates[-1], f"{self.dataset_name}_srgb_max_values.json")
        self.srgb_max_values = {}
        try:
            with open(srgb_max_json_path, 'r') as f:
                self.srgb_max_values = json.load(f)
        except FileNotFoundError:
            print(f"Warning: sRGB max values file not found at {srgb_max_json_path}. Scaling will not be applied for non-sRGB cameras.")

    def _load_hsi_data_from_hdf5(self, h5_file_handle):
        """Helper method to load HSI data from HDF5 and save to cache."""
        print(f"Loading {len(self._hsi_keys_to_load_in_memory)} HSI samples into memory for {self.dataset_name} - {self.dataset_type} set...")
        self.loaded_hsi_data = {}
        for hsi_key in tqdm(self._hsi_keys_to_load_in_memory, desc=f"Loading HSI data ({self.dataset_type})"):
            self.loaded_hsi_data[hsi_key] = h5_file_handle[hsi_key][:]
        
        print("HSI data loading into memory complete.")
        try:
            print(f"Saving HSI data to cache: {self.cache_file_path}")
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.loaded_hsi_data, f)
            print("HSI data saved to cache.")
        except Exception as e:
            print(f"Error saving to cache: {e}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Get the relative HSI path from metadata (e.g., "hs/image0001")
        hsi_relative_path = self.meta_data[index]['hsi']
        # Construct the full internal HDF5 path key used for memory lookup
        hsi_key_for_memory = f"{self.hsi_prefix}/{hsi_relative_path}"

        # Retrieve HSI data from memory
        hsi_data = self.loaded_hsi_data[hsi_key_for_memory]

        if self.camera_name == 'XYZ':
            target_lower_wavelength=451
            target_upper_wavelength=823
        else:
            target_lower_wavelength=451
            target_upper_wavelength=700

        input_lower_wavelength=451
        input_upper_wavelength=855
        spectrum_stepsize=13.3

        sliced_hsi, hsi_wavelength = slice_hsi_spectrum(
            hsi=hsi_data,
            input_lower_wavelength=input_lower_wavelength,
            input_upper_wavelength=input_upper_wavelength,
            target_lower_wavelength=target_lower_wavelength,
            target_upper_wavelength=target_upper_wavelength,
            spectrum_stepsize=spectrum_stepsize
        )

        hsi_tensor = torch.tensor(sliced_hsi, dtype=torch.float32).permute(2, 0, 1)
        target = int(self.targets[index])

        max_srgb = self.srgb_max_values.get(hsi_key_for_memory, 1.0)
        max_srgb = max(float(max_srgb), 1e-6)
        hsi_tensor = hsi_tensor / max_srgb
        hsi_tensor = torch.clamp(hsi_tensor, 0.0, 1.0)

        if self.transform:
            hsi_tensor = self.transform(hsi_tensor)

        return hsi_tensor, target

    def _get_hdf5_path(self):
        base = os.path.abspath(self.data_dir)
        if self.dataset_name == 'HFD100_Flower':
            return os.path.join(base, 'MatFlower60.h5')
        elif self.dataset_name == 'HFD100_Leaves':
            return os.path.join(base, 'MatLeaves60.h5')
        elif self.dataset_name == 'HFD100_Scenes':
            return os.path.join(base, 'MatScenes60.h5')
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")
    
    def num_classes(self):
        return len(set(self.targets))
    
    def _get_wavelengths(self):
        wavelengths = np.linspace(451, 855, 151)[::5]
        wavelengths = np.round(wavelengths).astype(int)
        return wavelengths
