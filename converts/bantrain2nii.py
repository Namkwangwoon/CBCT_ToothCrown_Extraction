import os
import json
import nibabel as nib
import numpy as np

JSON_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset')
DATA_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/Dataset')
SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/ban_nii')

ban_train = [
    'P20', 'P22', 'P23', 'P25', 'P33', 
    'P38', 'P60', 'P83', 'P84', 'P92',
    'P93', 'P102', 'P105', 'P107', 'P111', 
    'P119', 'P122', 'P131', 'P133', 'P134', 
    'P136', 'P152'
    ]

ban_test = [
    'P37', 'P139'
]
ban_val = [
    'P95', 'P128', 'P415'
]

total_ban = ban_train + ban_test + ban_val

save_types = ['train', 'test', 'val']
# save_types = ['val']

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

with open(os.path.join(JSON_PATH, 'splits.json'), 'r') as jr:
    json_data = json.load(jr)

for data_type in save_types:
    print('{} Data Lens : '.format(data_type), len(json_data[data_type]))

    save_dir = os.path.join(SAVE_PATH, data_type)

    for idx, file_name in enumerate(json_data[data_type]):
        print(idx, ' / ', len(json_data[data_type]))
        # if idx > 0:
        #     break

        if file_name in total_ban:
            continue

        data_dir = os.path.join(DATA_PATH, file_name)
        save_each_dir = os.path.join(save_dir, file_name)

        if not os.path.exists(save_each_dir):
            os.makedirs(save_each_dir)

        np_data = np.load(data_dir + '/data.npy')
        # np_sparse = np.load(data_dir + '/gt_sparse.npy')
        np_gt = np.load(data_dir + '/gt_alpha.npy')

        print('np_data.shape : ', np_data.shape)
        print('np_gt.shape : ', np_gt.shape)

        nii_data = nib.Nifti1Image(np_data, affine=np.eye(4))
        nii_gt = nib.Nifti1Image(np_gt, affine=np.eye(4))

        nib.save(nii_data, save_each_dir + '/data.nii.gz')
        nib.save(nii_gt, save_each_dir + '/gt_alpha.nii.gz')