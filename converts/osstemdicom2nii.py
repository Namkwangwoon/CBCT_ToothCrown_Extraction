import os
import json
import dicom2nifti
import nibabel as nib
import numpy as np

from skimage.measure import regionprops

import pydicom
from dcm2tensor import read_subject


DATA_PATH = os.path.abspath('../datasets/osstem_clean')
# SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/osstem_clean/2차/1/1/nii')

data_names = os.listdir(DATA_PATH)
data_names = sorted(data_names)
data_names = ['23']

for idx, data_name in enumerate(data_names):
    # if idx < 10:
    #     continue
    print('=> Progress : ', idx+1, ' / ', len(data_names))
    # if idx > 0:
    #     break
    cur_dir = os.path.join(DATA_PATH, data_name)

    '''
    dicom to nii
    '''

    dicom_dir = os.path.join(cur_dir, 'CT')
    save_dir = os.path.join(cur_dir, 'nii')

    # # if not os.path.exists(save_dir):
    # #     os.makedirs(save_dir)

    # # dicom2nifti.convert_directory(dicom_dir, save_dir)
    
    # org, img, voxel_spacing, initial_position = read_subject(os.listdir(dicom_dir), dicom_dir)
    # print(voxel_spacing)
    

    '''
    maskdata to nii
    '''

    mask_dir = os.path.join(cur_dir, 'Mask')
    mask_dir_file_names = sorted(os.listdir(mask_dir))

    mask_data_file_names = []
    whole_mask_arr = np.array([])
    box_dict = dict()

    for mask_file_name in mask_dir_file_names:
        if 'maskdata' not in mask_file_name:
            continue
        mask_data_file_names.append(mask_file_name)

    for idx2, mask_file_name in enumerate(mask_data_file_names):
    #     print('    Mask : ', idx2+1, ' / ', len(mask_data_file_names))
    #     # print('mask_file_name : ', mask_file_name)
        mask_idx = mask_file_name[:2]
        mask_info_name = '{}.mask'.format(mask_idx)

        try:
            if os.path.isfile(os.path.join(mask_dir, mask_info_name)):
                f = open(os.path.join(mask_dir, mask_info_name), 'r')
                line = f.readline().rstrip()
                line = line.replace(' ', '')
                h, w, d = line[:3], line[3:6], line[6:]
                d = int(d)
                w = int(w)
                h = int(h)
                f.close()
            else:
                continue

            mask_arr = np.fromfile(os.path.join(mask_dir, mask_file_name), dtype=np.int8)
            # print('mask_arr : ', mask_arr.shape)
            mask_arr = mask_arr.reshape((d, h, w))
        except Exception as e:
            continue

        # if whole_mask_arr.size == 0:
        #     whole_mask_arr = mask_arr
        # else:
        #     whole_mask_arr = whole_mask_arr + mask_arr

        # nii_mask_arr = nib.Nifti1Image(mask_arr, affine=np.eye(4))
        # nib.save(nii_mask_arr, save_dir + '/{}_gt.nii.gz'.format(mask_idx))

        img_array_for_box = np.asarray(mask_arr, dtype=int)
        props = regionprops(img_array_for_box)
        bbox = props[0].bbox
        box_dict[mask_idx] = bbox

    # nii_whole_mask_arr = nib.Nifti1Image(whole_mask_arr, affine=np.eye(4))
    # nib.save(nii_whole_mask_arr, save_dir + '/whole_mask.nii.gz')

    with open(save_dir + '/bbox.json','w') as f:
        json.dump(box_dict, f, indent=4)