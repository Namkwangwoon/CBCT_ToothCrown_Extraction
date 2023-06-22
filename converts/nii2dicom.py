import os
import json
import nibabel as nib
import numpy as np
import tqdm

import pydicom

DATA_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/nii_sparse/val/P24/data.nii.gz')
SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/dicom/val/P24')

def convertNsave(arr,file_dir=SAVE_PATH, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    
    dicom_file = pydicom.dcmread('/media/jeeheon/SSD/ToothFairy_Dataset/dicom/images/dcmimage.dcm', force=True)
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()

    dicom_file.save_as(os.path.join(file_dir, f'slice{index}.dcm'))

def nifti2dicom_1file(nifti_dir, out_dir=SAVE_PATH):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_file = nib.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    nifti_array = nifti_array.transpose(2, 1, 0)
    print('nifti_array : ', nifti_array.shape)
    number_slices = nifti_array.shape[2]

    # for slice_ in tqdm(range(number_slices)):
    for slice_ in range(number_slices):
        convertNsave(nifti_array[:,:,slice_], SAVE_PATH, slice_)

if __name__ == '__main__':
    nifti2dicom_1file(DATA_PATH)