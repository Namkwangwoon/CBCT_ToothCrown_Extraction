import os
import dicom2nifti
import nibabel as nib
import numpy as np

DATA_PATH = os.path.abspath('/media/jeeheon/SSD/Dataset_osstem_toothseg/2차/1/1/CT')
SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/Dataset_osstem_toothseg/2차/1/1/nii')

file_names = os.listdir(DATA_PATH)
file_names = sorted(file_names)

dicom2nifti.convert_directory(DATA_PATH, SAVE_PATH)