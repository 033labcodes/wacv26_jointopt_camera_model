import os
import numpy as np

def get_XYZ_CMFs(cmf_path: str=None):
    if cmf_path is None:
        _dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        cmf_path = os.path.join(_dir, 'CIE_XYZ_CMFs.csv')
    if not os.path.exists(cmf_path):
        raise FileNotFoundError(f"色度関数ファイルが見つかりません: {cmf_path}")
    cmf = np.genfromtxt(cmf_path, delimiter=',')
    return cmf

def hs_to_srgb(hsi: np.array, spectrum_stepsize: int=5):
    '''
    Parameters:
        hsi (np.array): hyperspectral image (height, width, band)(360mn~830nm)
        spectrum_stemsize (int): wavelength range between hsi channels

    Returns:
        np.array: NumPy array of RGB images converted from hyperspectral images
    '''

    hsi = hsi.astype(np.float32)
    height, width = hsi.shape[0], hsi.shape[1]

    color_matching_function = get_XYZ_CMFs()
    color_matching_function = color_matching_function[::spectrum_stepsize] 

    
    img_xyz = np.zeros((height, width, 3))
    img_rgb = np.zeros((height, width, 3))

    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0414],
                  [0.0557, -0.2040, 1.0570]])
    
    intensity = hsi.reshape(-1, hsi.shape[2])
    
    xyz = np.dot(intensity, color_matching_function[:, 1:])

    img_xyz = xyz.reshape(height, width, 3)

    img_rgb = np.dot(img_xyz, M.T)
    return img_rgb


if __name__ == "__main__":
    hsi = np.random.rand(100, 100, 95)
    rgb = hs_to_srgb(hsi)
    print(rgb.shape)
    
    import torch
    M = np.array([[3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0414],
                [0.0557, -0.2040, 1.0570]])
    M_tensor = torch.tensor(M, dtype=torch.float32)
    
    save_dir = os.path.join('../camera_parameters', 'ccm')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'ccm_sRGB.pt')
    torch.save(M_tensor, save_path)
    print(f'CCM行列を保存しました: {save_path}')
    
    # CIE XYZ CMFをテンソルとして保存
    cmf = get_XYZ_CMFs()
    cmf = cmf[::5]
    cmf_tensor = torch.tensor(cmf[:, 1:], dtype=torch.float32)
    cmf_tensor = cmf_tensor.transpose(0, 1)
    cmf_tensor = cmf_tensor.unsqueeze(-1).unsqueeze(-1)

    save_dir = os.path.join('../camera_parameters', 'css')
    cmf_save_path = os.path.join(save_dir, 'cmf_cie_XYZ.pt')
    torch.save(cmf_tensor, cmf_save_path)
    print(f'CIE XYZ CMFを保存しました: {cmf_save_path}')
