import os
import json
import nibabel as nib
import numpy as np

import zipfile


DATA_PATH = os.path.abspath('/media/jeeheon/SSD/osstem_clean')
SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/osstem_clean')

dir_names = os.listdir(DATA_PATH)
dir_names = sorted(dir_names)

cnt = 0

for dir_name in dir_names:
    cur_path = os.path.join(DATA_PATH, dir_name)
    file_names = os.listdir(cur_path)
    file_names = sorted(file_names)

    for file_name in file_names:
        print('PATH : ', os.path.join(cur_path, file_name))
        if '.zip' in file_name:
            with zipfile.ZipFile(os.path.join(cur_path, file_name), 'r') as zip_ref:
                zip_ref.extractall(cur_path)
            cnt += 1

    # if cnt > 0:
    #     break