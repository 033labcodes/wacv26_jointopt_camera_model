import os
import numpy as np
import torch
from scipy import interpolate


def slice_spectrum(spectrum: np.array, input_lower_wavelength: int, input_upper_wavelength: int,
                  target_lower_wavelength: int, target_upper_wavelength: int, spectrum_stepsize: float):
    """Slice spectrum by wavelength range. Returns (sliced_spectrum, wavelengths)."""
    input_wavelength = np.array(np.arange(input_lower_wavelength, input_upper_wavelength + 1, spectrum_stepsize), dtype=int)
    start_index = np.argmin(np.abs(input_wavelength - target_lower_wavelength))
    end_index = np.argmin(np.abs(input_wavelength - target_upper_wavelength)) + 1
    return spectrum[start_index:end_index], input_wavelength[start_index:end_index]


def adjust_cmf_to_target_wavelength(cmf_array, source_wavelength, target_wavelength):
    """Interpolate CMF to target wavelengths. Returns adjusted CMF array."""
    if cmf_array.shape[1] != len(source_wavelength):
        print(f"Warning: Mismatch in array lengths. cmf_array: {cmf_array.shape[1]}, source_wavelength: {len(source_wavelength)}")
        # Truncate to shorter length
        min_length = min(cmf_array.shape[1], len(source_wavelength))
        cmf_array = cmf_array[:, :min_length]
        source_wavelength = source_wavelength[:min_length]
    
    adjusted_cmf = np.zeros((cmf_array.shape[0], len(target_wavelength)))
    
    for i in range(cmf_array.shape[0]):
        interp_func = interpolate.interp1d(
            source_wavelength, 
            cmf_array[i], 
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        adjusted_cmf[i] = interp_func(target_wavelength)
    
    return adjusted_cmf


def convert_css_wavelength_to_tensor(input_dir, output_dir,
                                   input_lower_wavelength=400, input_upper_wavelength=700,
                                   target_lower_wavelength=451, target_upper_wavelength=700,
                                   spectrum_stepsize=9.357):
    """Adjust CSS wavelength range and convert to tensor."""
    os.makedirs(output_dir, exist_ok=True)
    
    # HSI wavelength range
    hsi_wavelength = np.arange(target_lower_wavelength, target_upper_wavelength + 5, 13.3)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy') and not file_name.startswith('.'):
            file_path = os.path.join(input_dir, file_name)
            array = np.load(file_path)
            if array.shape[1] != 33:
                print(f"{file_name} - Warning - array shape: {array.shape}")
                continue
            
            # Slice wavelength range
            sliced_arrays = []
            for i in range(array.shape[0]):
                spectrum = array[i, :].reshape(-1)
                sliced_array, css_wavelength = slice_spectrum(
                    spectrum=spectrum,
                    input_lower_wavelength=input_lower_wavelength,
                    input_upper_wavelength=input_upper_wavelength,
                    target_lower_wavelength=target_lower_wavelength,
                    target_upper_wavelength=target_upper_wavelength,
                    spectrum_stepsize=spectrum_stepsize
                )
                sliced_arrays.append(sliced_array)
            
            sliced_array = np.array(sliced_arrays)
            
            # Interpolate to HSI wavelengths
            adjusted_array = adjust_cmf_to_target_wavelength(
                sliced_array,
                css_wavelength,
                hsi_wavelength
            )
            
            # Convert to tensor
            tensor = torch.from_numpy(adjusted_array.reshape(3, -1, 1, 1))
            
            output_path = os.path.join(output_dir, file_name.replace('.npy', '.pt'))
            torch.save(tensor, output_path)
            print(f'Saved: {file_name} -> {os.path.basename(output_path)}, shape: {tensor.shape}')


if __name__ == '__main__':
    input_dir = '/mnt/hdd3/youta/ws/optimize_css_flower/src/camera_parameters/dimention_33'
    output_dir = '/mnt/hdd3/youta/ws/optimize_css_flower/src/camera_parameters/css'
    convert_css_wavelength_to_tensor(input_dir, output_dir)