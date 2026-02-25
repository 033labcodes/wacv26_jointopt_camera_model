import numpy as np

def slice_hsi_spectrum(hsi: np.array, input_lower_wavelength: int, input_upper_wavelength: int, target_lower_wavelength: int, target_upper_wavelength, spectrum_stepsize: float):
    input_wavelength = np.array(np.arange(input_lower_wavelength, input_upper_wavelength + 1, spectrum_stepsize), dtype=int)

    start_index = np.argmin(np.abs(input_wavelength - target_lower_wavelength))
    end_index = np.argmin(np.abs(input_wavelength - target_upper_wavelength)) + 1    
    return hsi[:, :, start_index:end_index], input_wavelength[start_index:end_index]

def slice_spectrum(spectrum: np.array, input_lower_wavelength: int, input_upper_wavelength: int, target_lower_wavelength: int, target_upper_wavelength, spectrum_stepsize: float):
    input_wavelength = np.array(np.arange(input_lower_wavelength, input_upper_wavelength + 1, spectrum_stepsize), dtype=int)

    start_index = np.argmin(np.abs(input_wavelength - target_lower_wavelength))
    end_index = np.argmin(np.abs(input_wavelength - target_upper_wavelength)) + 1
    return spectrum[start_index:end_index], input_wavelength[start_index:end_index]

def slice_wavelength_and_intensity(wavelength, intensity, start: int=360, end: int=830, step: int=5):
    start_idx = np.where(wavelength >= start)[0][0]
    end_idx = np.where(wavelength <= end)[0][-1]
    wavelength = wavelength[start_idx:end_idx + 1:step]
    intensity = intensity[start_idx:end_idx + 1:step]
    return wavelength, intensity