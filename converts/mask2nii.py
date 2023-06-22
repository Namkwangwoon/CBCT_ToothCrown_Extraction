import os
import json
import nibabel as nib
import numpy as np

# JSON_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset')
DATA_PATH = os.path.abspath('/media/jeeheon/SSD/Dataset_osstem_toothseg/2차/1/1/Mask')
TEMP_PATH = os.path.abspath('/media/jeeheon/SSD/Dataset_osstem_toothseg/2차/1/1')
SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/nii_sparse')

# with open(os.path.join(JSON_PATH, 'splits.json'), 'r') as jr:
#     json_data = json.load(jr)

# save_dir = os.path.join(SAVE_PATH, data_type)

file_names = os.listdir(DATA_PATH)
file_names = sorted(file_names)

for idx, file_name in enumerate(file_names):
    print(idx, ' / ', len(file_names), ' => ', file_name)
    if idx > 1:
        break

    if '.maskdata' not in file_name:
        print('BAN => ', file_name)
        continue
        # break
    # file_name = '11.raw'

    # data_dir = os.path.join(DATA_PATH, file_name)
    # save_each_dir = os.path.join(save_dir, file_name)

    # if not os.path.exists(save_each_dir):
    #     os.makedirs(save_each_dir)

    # np_data = np.load(data_dir + '/data.npy')
    # np_gt = np.load(data_dir + '/gt_sparse.npy')

    # np_gt = np.fromfile(DATA_PATH + '/' + file_name, dtype=np.byte, like=np.array([752, 752, 450]))
    np_gt = np.fromfile(DATA_PATH + '/' + file_name, dtype=np.int8)
    # np_gt = np.fromfile(TEMP_PATH + '/' + file_name, dtype=np.int8)
    # np_gt = np.fromfile(DATA_PATH + '/' + file_name, dtype=np.int8, like=np.array([752, 752, 450]))

    # print('np_data.shape : ', np_data.shape)
    print('np_gt.shape : ', np_gt.shape)
    # np_gt = np_gt.reshape((752, 752, 450))
    np_gt = np_gt.reshape((450, 752, 752))

    # nii_data = nib.Nifti1Image(np_data, affine=np.eye(4))
    nii_gt = nib.Nifti1Image(np_gt, affine=np.eye(4))
    nib.save(nii_gt, './gt_from_maskdata_temp.nii.gz')

    # nib.save(nii_data, save_each_dir + '/data.nii.gz')
    # nib.save(nii_gt, save_each_dir + '/gt_sparse.nii.gz')