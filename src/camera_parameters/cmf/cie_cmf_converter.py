import numpy as np
import torch
from scipy import interpolate


def read_cie1931_cmf(csv_path):
    """
    CIE1931 XYZ CMFデータを読み込む
    
    Args:
        csv_path (str): CSVファイルのパス
        
    Returns:
        tuple: (波長配列, CMF配列)
    """
    data = np.loadtxt(csv_path, delimiter=',')
    wavelength = data[:, 0]
    cmf = data[:, 1:4].T  # 転置して[3, n]の形状に
    
    return wavelength, cmf


def adjust_cmf_to_target_wavelength(cmf_array, source_wavelength, target_wavelength):
    """
    色マッチング関数を目標波長に合わせて補間する
    
    Args:
        cmf_array (np.ndarray): 色マッチング関数の配列（shape: [3, n]）
        source_wavelength (np.ndarray): 元の波長の配列
        target_wavelength (np.ndarray): 目標の波長の配列
        
    Returns:
        np.ndarray: 補間されたCMF配列
    """    
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


def slice_cmf_spectrum(cmf_array, wavelength, target_lower_wavelength=451, target_upper_wavelength=830, spectrum_stepsize=13.3):
    """
    CMFデータを指定された波長範囲でスライスする
    
    Args:
        cmf_array (np.ndarray): CMF配列（shape: [3, n]）
        wavelength (np.ndarray): 波長配列
        target_lower_wavelength (int): 目標波長の下限
        target_upper_wavelength (int): 目標波長の上限
        spectrum_stepsize (float): 波長のステップサイズ
        
    Returns:
        tuple: (スライスされたCMF配列, 対応する波長)
    """
    target_wavelength = np.arange(target_lower_wavelength, target_upper_wavelength + 1, spectrum_stepsize)
    start_index = np.argmin(np.abs(wavelength - target_lower_wavelength))
    end_index = np.argmin(np.abs(wavelength - target_upper_wavelength)) + 1
    
    # 波長に合わせて補間
    adjusted_cmf = adjust_cmf_to_target_wavelength(
        cmf_array,
        wavelength,
        target_wavelength
    )
    
    return adjusted_cmf, target_wavelength


def convert_cie_cmf_to_tensor(csv_path, output_path, 
                            target_lower_wavelength=451, target_upper_wavelength=830,
                            spectrum_stepsize=13.3):
    """
    CIE1931 XYZ CMFデータを指定された波長域で補間し，テンソルに変換する
    
    Args:
        csv_path (str): 入力CSVファイルのパス
        output_path (str): 出力ファイルのパス
        target_lower_wavelength (int): 目標波長の下限
        target_upper_wavelength (int): 目標波長の上限
        spectrum_stepsize (float): 波長のステップサイズ
    """
    # CMFデータの読み込み
    wavelength, cmf = read_cie1931_cmf(csv_path)
    
    # 波長範囲のスライスと補間
    sliced_cmf, target_wavelength = slice_cmf_spectrum(
        cmf,
        wavelength,
        target_lower_wavelength,
        target_upper_wavelength,
        spectrum_stepsize
    )
    
    print(f'目標波長: {target_wavelength}')
    print(f'波長数: {len(target_wavelength)}')
    
    # テンソルに変換
    tensor = torch.from_numpy(sliced_cmf.reshape(3, -1, 1, 1))
    
    # 保存
    torch.save(tensor, output_path)
    print(f'変換して保存しました: {output_path}')
    print(f'変換後の形状: {tensor.shape}')


if __name__ == '__main__':
    csv_path = './cie1931_xyz_cmf.csv'
    output_path = './cie1931_xyz_cmf.pt'
    convert_cie_cmf_to_tensor(csv_path, output_path)
