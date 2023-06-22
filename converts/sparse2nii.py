import os
import json
import nibabel as nib
import numpy as np

JSON_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset')
DATA_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/Dataset')
SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/nii_sparse')

data_types = ['test', 'val']

with open(os.path.join(JSON_PATH, 'splits.json'), 'r') as jr:
    json_data = json.load(jr)


for data_type in data_types:
    print('{} Data Lens : '.format(data_type), len(json_data[data_type]))

    save_dir = os.path.join(SAVE_PATH, data_type)

    for idx, file_name in enumerate(json_data[data_type]):
        print(idx, ' / ', len(json_data[data_type]))
        # if idx > 6:
        #     break

        data_dir = os.path.join(DATA_PATH, file_name)
        save_each_dir = os.path.join(save_dir, file_name)

        if not os.path.exists(save_each_dir):
            os.makedirs(save_each_dir)

        np_data = np.load(data_dir + '/data.npy')
        np_gt = np.load(data_dir + '/gt_sparse.npy')

        print('np_data.shape : ', np_data.shape)
        print('np_gt.shape : ', np_gt.shape)

        nii_data = nib.Nifti1Image(np_data, affine=np.eye(4))
        nii_gt = nib.Nifti1Image(np_gt, affine=np.eye(4))

        nib.save(nii_data, save_each_dir + '/data.nii.gz')
        nib.save(nii_gt, save_each_dir + '/gt_sparse.nii.gz')
    # break