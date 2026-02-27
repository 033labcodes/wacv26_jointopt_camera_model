import numpy as np

class SpectrumAdjuster:
    def __init__(self, lower_limit_wavelength, upper_limit_wavelength, spectrum_stepsize):
        self.current_wavelengths = np.linspace(451, 855, 31)
        self.lower_limit_wavelength = lower_limit_wavelength
        self.upper_limit_wavelength = upper_limit_wavelength
        self.spectrum_stepsize = spectrum_stepsize
        self._adjust_wavelengths()
        
    def _adjust_wavelengths(self):
        min_current = self.current_wavelengths[0]
        max_current = self.current_wavelengths[-1]
        
        lower_extension = np.arange(self.lower_limit_wavelength, min_current, self.spectrum_stepsize)
        upper_extension = np.arange(max_current + self.spectrum_stepsize, self.upper_limit_wavelength + self.spectrum_stepsize, self.spectrum_stepsize)
        
        new_wavelengths = np.concatenate([lower_extension, self.current_wavelengths, upper_extension])
    
        mask = (new_wavelengths >= self.lower_limit_wavelength) & (new_wavelengths <= self.upper_limit_wavelength)
        new_wavelengths = new_wavelengths[mask]
        self.new_wavelengths = new_wavelengths

    def transform_hsi_data(self, hsi_data):
        """Adjust HSI spectral bands. Args: hsi_data (H,W,bands). Returns: adjusted HSI, wavelengths."""
        height, width = hsi_data.shape[:2]

        adjusted_data = np.zeros((height, width, len(self.new_wavelengths)))
        
        mask = np.isin(self.new_wavelengths, self.current_wavelengths)
        current_indices = np.searchsorted(self.current_wavelengths, self.new_wavelengths[mask])
        
        adjusted_data[:, :, mask] = hsi_data[:, :, current_indices]
        return adjusted_data