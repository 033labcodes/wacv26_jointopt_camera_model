import numpy as np
import pandas as pd
from pathlib import Path

def load_CIE_std_illum_D65(path):
    """
    CIE標準光源D65のデータを読み込む関数
    
    Returns:
        tuple: (wavelength, intensity)
            - wavelength: 波長配列 (nm)
            - intensity: 相対強度配列
    """
    df = pd.read_csv(path, header=None, names=["wavelength", "intensity"])
    wavelength = df["wavelength"].values
    intensity = df["intensity"].values
    
    return wavelength, intensity

if __name__ == "__main__":
    path = "/mnt/hdd1/youta/ws/optimize_css/data/CIE_std_illum_D65.csv"
    wavelength, intensity = load_CIE_std_illum_D65(path)
    
    from slice_wavelength import slice_wavelength_and_intensity
    wavelength, intensity = slice_wavelength_and_intensity(wavelength, intensity)
    print(wavelength.shape)
    print(intensity.shape)
