import json
import h5py
import numpy as np

def convert_illum(illuminant, refrectance):
    converted_hsi = illuminant * refrectance
    return converted_hsi

def get_reflectance(hsi, white_spectra):
    reflectance = hsi / white_spectra
    return reflectance



def main(dataset_info_list, white_board_spectra, dataset_h5_path, illuminant_intensity):
    h5_file = h5py.File(dataset_h5_path, 'r')
    new_dataset_info_list = []
    for i, dataset_info in enumerate(dataset_info_list):
        print(f'{i}/{len(dataset_info_list)}')
        if dataset_info['meta_data']['white_image_id'] == None:
            continue
        
        image_id = dataset_info['image_id']
        white_image_id = dataset_info['meta_data']['white_image_id']

        white_spectra = white_board_spectra[f'white/{white_image_id}']['average_spectrum']
        white_spectra = np.array(white_spectra)
        white_spectra = slice_spectrum(white_spectra, lower_limit_wavelength=360, upper_limit_wavelength=830, spectrum_stepsize=5)
        
        hsi = h5_file[f'hsi/{image_id}'][:]
        hsi = slice_hsi_spectrum(hsi, lower_limit_wavelength=360, upper_limit_wavelength=830, spectrum_stepsize=5)
        
        reflectance = get_reflectance(hsi, white_spectra)
        
        converted_hsi = convert_illum(illuminant_intensity, reflectance)
        converted_rgb = hs_to_srgb(converted_hsi)
        threshold = np.max(converted_rgb)

        converted_hsi_normalized = converted_hsi / threshold
        converted_rgb_normalized = hs_to_srgb(converted_hsi_normalized)

        converted_rgb_normalized = np.clip(converted_rgb_normalized, 0, 1)
        converted_rgb_normalized = converted_rgb_normalized ** (1/2.2)
        converted_rgb_normalized = (converted_rgb_normalized * 255).astype(np.uint8)
        converted_rgb_normalized_bgr = cv2.cvtColor(converted_rgb_normalized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'/mnt/hdd1/youta/ws/optimize_css/data/demo_rgbs_2/{image_id}.png', converted_rgb_normalized_bgr)
        
        new_dataset_info = dataset_info.copy()
        new_dataset_info['meta_data']['threshold'] = threshold
        print(threshold)
        new_dataset_info_list.append(new_dataset_info)
    return new_dataset_info_list



if __name__ == "__main__":
    import cv2
    from load_CIE_std_illum import load_CIE_std_illum_D65
    from hs_to_srgb import hs_to_srgb
    from slice_wavelength import slice_hsi_spectrum, slice_spectrum, slice_wavelength_and_intensity
    
    # dataset_info_list_path = '/mnt/hdd1/youta/ws/optimize_css/data/hyper_penguin.json'
    dataset_info_list_path = '/mnt/hdd1/youta/ws/optimize_css/data/val.json'
    white_board_spectra_path = '/mnt/hdd1/youta/ws/optimize_css/data/white_spectra.json'
    dataset_h5_path = '/mnt/hdd1/datasets/hyperspectral/hyper_penguin/hyper_penguin_4/hyper_penguin.h5'
    illuminant_path = '/mnt/hdd1/youta/ws/optimize_css/data/CIE_std_illum_D65.csv'
    illuminant_wavelength, illuminant_intensity = load_CIE_std_illum_D65(illuminant_path)
    illuminant_wavelength, illuminant_intensity = slice_wavelength_and_intensity(illuminant_wavelength, illuminant_intensity, start=360, end=830, step=5)

    dataset_info_list = json.load(open(dataset_info_list_path))
    white_board_spectra = json.load(open(white_board_spectra_path))
    new_dataset_info_list = main(dataset_info_list, white_board_spectra, dataset_h5_path, illuminant_intensity)
    save_path = '/mnt/hdd1/youta/ws/optimize_css/data/val_converted.json'
    json.dump(new_dataset_info_list, open(save_path, 'w'), indent=4)
    print(f'{save_path} done')