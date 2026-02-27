# compute_srgb_max.py

import h5py
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import torch # For loading .pt file

# --- Helper function (assuming it might be used or adapted if needed, or replaced by direct slicing logic) ---
def slice_hsi_spectrum(hsi: np.array, input_lower_wavelength: int, input_upper_wavelength: int, target_lower_wavelength: int, target_upper_wavelength, spectrum_stepsize: float):
    input_wavelength_grid = np.array(np.arange(input_lower_wavelength, input_upper_wavelength + 1, spectrum_stepsize), dtype=float)
    start_index = np.argmin(np.abs(input_wavelength_grid - target_lower_wavelength))
    end_index = np.argmin(np.abs(input_wavelength_grid - target_upper_wavelength)) + 1
    sliced_hsi = hsi[:, :, start_index:end_index]
    sliced_wavelengths = input_wavelength_grid[start_index:end_index]
    return sliced_hsi, sliced_wavelengths
# --- End helper function ---

def hsi_to_srgb(hsi_data, cmfs):
    """
    Converts HSI data to sRGB image, assuming HSI data is ready for CMF multiplication.
    Args:
        hsi_data (np.ndarray): Sliced HSI data cube (H, W, C_sliced).
        cmfs (np.ndarray): Color Matching Functions (C_sliced, 3) matched to HSI data channels.
    Returns:
        np.ndarray: sRGB image (H, W, 3), values in [0, 1]. None if conversion fails.
    """
    H, W, C_sliced = hsi_data.shape
    if C_sliced != cmfs.shape[0]:
        print(f"Error: Dimension mismatch. HSI_C={C_sliced}, CMF_rows={cmfs.shape[0]}")
        return None

    # Directly calculate XYZ-like values by multiplying HSI data with CMFs
    # This assumes hsi_data is in a form (e.g., radiance under a standard illuminant, or already D65-weighted reflectance)
    # such that hsi_data @ cmfs yields XYZ or a value proportional to it.
    hsi_flat = hsi_data.reshape((-1, C_sliced))
    xyz_flat = hsi_flat @ cmfs # (H*W, C_sliced) @ (C_sliced, 3) -> (H*W, 3)
    xyz_image = xyz_flat.reshape((H, W, 3))

    M_xyz_to_rgb = np.array([ # Standard sRGB D65 matrix
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    linear_rgb_flat = xyz_flat @ M_xyz_to_rgb.T
    linear_rgb_image = linear_rgb_flat.reshape((H, W, 3))

    return linear_rgb_image

def load_srgb_cmfs_from_pt(pt_file_path, expected_hsi_channels):
    print(f"--- Inside load_srgb_cmfs_from_pt ---")
    print(f"Loading CMF from: {pt_file_path}")
    print(f"Expected HSI channels for CMF: {expected_hsi_channels}")

    if not os.path.exists(pt_file_path):
        print(f"Error: CMF .pt file not found at {pt_file_path}")
        return None
    try:
        cmf_tensor = torch.load(pt_file_path, map_location=torch.device('cpu'))
        
        if not isinstance(cmf_tensor, torch.Tensor):
            print(f"Error: Content of {pt_file_path} is not a torch.Tensor. Found type: {type(cmf_tensor)}")
            return None
            
        print(f"Original CMF tensor loaded. Shape: {cmf_tensor.shape}, Dtype: {cmf_tensor.dtype}, Device: {cmf_tensor.device}")
        
        cmfs_np = cmf_tensor.numpy()
        print(f"CMF converted to NumPy. Original shape: {cmfs_np.shape}, Dtype: {cmfs_np.dtype}")

        # Squeeze out dimensions of size 1
        squeezed_cmfs_np = np.squeeze(cmfs_np)
        print(f"CMF after squeeze. Shape: {squeezed_cmfs_np.shape}")

        processed_cmfs_np = None
        if squeezed_cmfs_np.ndim == 2:
            if squeezed_cmfs_np.shape[0] == 3 and squeezed_cmfs_np.shape[1] == expected_hsi_channels:
                processed_cmfs_np = squeezed_cmfs_np.T
                print(f"Squeezed CMF shape was (3, {expected_hsi_channels}). Transposed to: {processed_cmfs_np.shape}")
            elif squeezed_cmfs_np.shape[0] == expected_hsi_channels and squeezed_cmfs_np.shape[1] == 3:
                processed_cmfs_np = squeezed_cmfs_np
                print(f"Squeezed CMF shape was ({expected_hsi_channels}, 3). Used as is: {processed_cmfs_np.shape}")
            else:
                print(f"Error: Squeezed CMF shape {squeezed_cmfs_np.shape} is 2D but does not match expected patterns: (3, {expected_hsi_channels}) or ({expected_hsi_channels}, 3).")
                return None
        elif squeezed_cmfs_np.ndim == 1 and expected_hsi_channels == 1 and squeezed_cmfs_np.shape[0] == 3:
            # Special case: if expected channels is 1 (e.g. grayscale) and CMF is (3,)
            # This might not be the case here given expected_hsi_channels=29, but as a robust check.
            processed_cmfs_np = squeezed_cmfs_np.reshape((1,3)) # Make it (1,3) for (H*W,1) @ (1,3)
            print(f"Squeezed CMF was 1D ({squeezed_cmfs_np.shape}), reshaped to {processed_cmfs_np.shape} for expected_hsi_channels=1.")
        else:
            print(f"Error: Squeezed CMF from {pt_file_path} is not a 2D array compatible with expectations. Actual ndim after squeeze: {squeezed_cmfs_np.ndim}, Shape: {squeezed_cmfs_np.shape}")
            return None
        
        if processed_cmfs_np is not None:
             print(f"Successfully processed CMF. Final shape to be used: {processed_cmfs_np.shape}")
        print(f"-------------------------------------")
        return processed_cmfs_np
        
    except Exception as e:
        print(f"Error loading or processing CMF .pt file {pt_file_path}: {e}")
        print(f"-------------------------------------")
        return None

def get_hdf5_path(dataset_name, data_dir):
    base = os.path.abspath(data_dir)
    if dataset_name == 'HFD100_Flower':
        return os.path.join(base, 'MatFlower60.h5')
    elif dataset_name == 'HFD100_Leaves':
        return os.path.join(base, 'MatLeaves60.h5')
    elif dataset_name == 'HFD100_Scenes':
        return os.path.join(base, 'MatScenes60.h5')
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

def main(dataset_name, output_json_path, data_dir, cmf_pt_path):
    hdf5_path = get_hdf5_path(dataset_name, data_dir)

    if not os.path.exists(hdf5_path) or not os.path.exists(cmf_pt_path):
        print(f"Error: HDF5 file ({hdf5_path}) or CMF file ({cmf_pt_path}) not found.")
        return

    print(f"Processing dataset: {dataset_name} from {hdf5_path}")
    print(f"Using CMF from: {cmf_pt_path}")
    print(f"Output JSON: {output_json_path}")

    srgb_max_values = {}

    # Slicing parameters for sRGB camera simulation (mirroring hsi_dataset.py for 'sRGB')
    slicing_input_lower_wl = 451
    slicing_input_upper_wl = 855
    slicing_step_size = 13.3 # Defines the grid for slice_hsi_spectrum's internal reference
    srgb_target_lower_wl = 451
    srgb_target_upper_wl = 823

    # Determine expected channels after slicing (for CMF validation)
    _ref_wl_grid = np.array(np.arange(slicing_input_lower_wl, slicing_input_upper_wl + 1, slicing_step_size), dtype=float)
    _start_idx_ref = np.argmin(np.abs(_ref_wl_grid - srgb_target_lower_wl))
    _end_idx_ref = np.argmin(np.abs(_ref_wl_grid - srgb_target_upper_wl)) + 1
    expected_sliced_channels = _end_idx_ref - _start_idx_ref
    print(f"Expecting {expected_sliced_channels} channels after slicing for sRGB simulation.")

    cmfs_srgb_sliced = load_srgb_cmfs_from_pt(cmf_pt_path, expected_sliced_channels)
    if cmfs_srgb_sliced is None:
        print("Failed to load CMFs. Aborting.")
        return

    try:
        with h5py.File(hdf5_path, 'r') as h5_file:
            for group_name in ['train', 'test']:
                if group_name not in h5_file:
                    continue
                print(f"\nProcessing group: {group_name}")
                metadata_key = f'{group_name}/metadata'
                if metadata_key not in h5_file:
                    continue
                metadata = json.loads(h5_file[metadata_key][()])

                for item in tqdm(metadata, desc=f"Calculating sRGB max ({group_name})"):
                    hsi_meta_path = item.get('hsi')
                    if not hsi_meta_path:
                        continue
                    hsi_full_path = f"{group_name}/{hsi_meta_path}"
                    if hsi_full_path not in h5_file:
                        continue

                    try:
                        original_hsi_data = h5_file[hsi_full_path][:]
                    except Exception as e:
                        print(f"Error loading original HSI data {hsi_full_path}: {e}. Skipping.")
                        continue
                    
                    if original_hsi_data.ndim != 3:
                        continue

                    # Slice original HSI data to simulate sRGB camera response
                    # 1. Define wavelengths of the original HSI (e.g., 151 bands from 451-855nm)
                    original_hsi_wavelengths = np.linspace(slicing_input_lower_wl, slicing_input_upper_wl, original_hsi_data.shape[2])
                    
                    # 2. Define target wavelengths for the sRGB camera simulation (approx. `expected_sliced_channels` bands)
                    #    These correspond to the CMF from `cie1931_xyz_cmf.pt`.
                    #    Generated using the same slicing logic as in hsi_dataset.py.
                    srgb_simulation_ref_grid = np.array(np.arange(slicing_input_lower_wl, slicing_input_upper_wl + 1, slicing_step_size), dtype=float)
                    s_idx_sim = np.argmin(np.abs(srgb_simulation_ref_grid - srgb_target_lower_wl))
                    e_idx_sim = np.argmin(np.abs(srgb_simulation_ref_grid - srgb_target_upper_wl)) + 1
                    target_wavelengths_for_srgb_sim = srgb_simulation_ref_grid[s_idx_sim:e_idx_sim]

                    if len(target_wavelengths_for_srgb_sim) != expected_sliced_channels:
                        # This might occur if np.arange behavior with float steps causes slight differences
                        print(f"Warning: Calculated target wavelengths ({len(target_wavelengths_for_srgb_sim)}) differs from initially expected ({expected_sliced_channels}). Using calculated length for safety checks.")
                        # It's crucial that the CMF dimensions match the actual sliced data dimensions.

                    # 3. For each target sRGB simulation wavelength, find the closest band in the original HSI.
                    selected_indices_from_original = [np.argmin(np.abs(original_hsi_wavelengths - t_wl)) for t_wl in target_wavelengths_for_srgb_sim]
                    
                    # 4. Select these bands from the original HSI data.
                    hsi_data_for_srgb_calc = original_hsi_data[:, :, selected_indices_from_original]

                    # Validate that the sliced data has the same number of channels as the CMF expects
                    if hsi_data_for_srgb_calc.shape[2] != cmfs_srgb_sliced.shape[0]: # CMF is (channels, 3)
                         print(f"Warning: Channel count of sliced HSI ({hsi_data_for_srgb_calc.shape[2]}) does not match CMF channels ({cmfs_srgb_sliced.shape[0]}). Skipping {hsi_full_path}.")
                         continue
                                        
                    srgb_image = hsi_to_srgb(hsi_data_for_srgb_calc, cmfs_srgb_sliced)

                    if srgb_image is not None:
                        max_val = np.max(srgb_image)
                        srgb_max_values[hsi_full_path] = float(max_val)
                    else:
                        print(f"Warning: sRGB conversion function returned None for {hsi_full_path}. Skipping.")

    except Exception as e:
        print(f"\nAn error occurred during HDF5 processing: {e}")
        return

    if not srgb_max_values:
         print("\nWarning: No sRGB max values were computed. JSON file will not be created.")
         return
    try:
        with open(output_json_path, 'w') as f:
            json.dump(srgb_max_values, f, indent=4)
        print(f"\nSuccessfully saved {len(srgb_max_values)} sRGB max values to {output_json_path}")
    except Exception as e:
        print(f"\nError saving JSON file: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_cmf = os.path.join(script_dir, '..', 'src', 'camera_parameters', 'css', 'cmf_XYZ.pt')
    
    parser = argparse.ArgumentParser(description="Compute maximum sRGB values for HSI datasets by simulating sRGB camera slicing.")
    parser.add_argument("dataset_name", choices=['HFD100_Flower', 'HFD100_Leaves', 'HFD100_Scenes'], help="Name of the HFD100 dataset to process.")
    parser.add_argument("-o", "--output", default=None, help="Path to save the output JSON file. Defaults to '{dataset_name}_srgb_max_values.json'.")
    parser.add_argument("-d", "--data-dir", default=os.environ.get('HFD100_DATA_DIR', './data'), help="Directory containing Mat*.h5 files. Default: HFD100_DATA_DIR or ./data")
    parser.add_argument("--cmf-path", default=default_cmf, help="Path to cmf_XYZ.pt. Default: src/camera_parameters/css/cmf_XYZ.pt")
    args = parser.parse_args()
    
    output_file = args.output if args.output else os.path.join(script_dir, f"{args.dataset_name}_srgb_max_values.json")
    cmf_path = os.path.abspath(os.path.expanduser(args.cmf_path))
    main(args.dataset_name, output_file, args.data_dir, cmf_path)