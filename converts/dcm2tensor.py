""" Converts the directory of DICOM files to a directory of .npy files
    Writes out a JSON file mapping subject ID to 3D voxel spacing
"""
import json
import sys
import torch
import torch.nn.functional as F
from operator import itemgetter
import os
from natsort import natsorted

import pydicom
import numpy as np

# Directory of original data

def resizing_windowing(img, mode='bilinear', windowding=True, normalization=True, depth=64, height=128, width=128):

    # Resizing
    d = torch.linspace(-1,1,depth)
    h = torch.linspace(-1,1,height)
    w = torch.linspace(-1,1,width)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshx, meshy, meshz), 3) # (64, 128, 128, 3(x,y,z))
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)

    img = torch.Tensor(img) # 
    img = img.unsqueeze(0).unsqueeze(0).permute(0,1,4,3,2)

    out = F.grid_sample(img, grid, mode=mode, align_corners=True)
    out = out.permute(0,1,4,3,2)[0,0,:,:,:]

    # windowing
    if windowding :
        window_level = 10 # 50
        window_width = 700 # 400
        window_high = (window_width / 2) + window_level
        window_low = window_level - (window_width / 2)

        out = out.clip(window_low, window_high)

    # Normalization
    if normalization:
        out -= torch.min(out)
        out /= (torch.max(out) - torch.min(out))
    return out

def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos

def read_subject(subjects, dir_path):
    """ Read in the directory of a single subject and return a numpy array """
            
    files = [os.path.join(dir_path, fname)
            for fname in subjects if fname.endswith('.dcm') and not fname.startswith('RS.')]
    
    header_files = [os.path.join(dir_path, fname)
            for fname in subjects if fname.startswith('RS.')]

    # if len(header_files) != 1 or len(header_files) != 0:
    #     sys.exit('Only one DCM files please')

    # Read slices as a list before sorting
    dcm_slices = [pydicom.read_file(fname) for fname in files]

    # Extract position for each slice to sort and calculate slice spacing
    dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcm_slices]
    dcm_slices = natsorted(dcm_slices, key=itemgetter(1)) # list of tuple : [(dcm, z-axis position), ...]

    initial_patient_position = dcm_slices[0][0].ImagePositionPatient # (w, h, d) position

    spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
    slice_spacing = np.mean(spacings)

    # All slices will have the same in-plane shape
    shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
    nslices = len(dcm_slices)

    # Final 3D array will be N_Slices x Columns x Rows
    shape = (nslices, *shape)
    img = np.empty(shape, dtype='float32')
    org = np.empty(shape, dtype='float32')
    
    for idx, (dcm, _) in enumerate(dcm_slices):
        # Rescale and shift in order to get accurate pixel values
        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)
        org[idx, ...] = dcm.pixel_array
        img[idx, ...] = dcm.pixel_array.astype('float32')*slope + intercept

    # Calculate size of a voxel in mm
    pixel_spacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)
    voxel_spacing = (slice_spacing, *pixel_spacing)

    return org, img, voxel_spacing, initial_patient_position

def dcm2tensor(dir_path):
    """ Converts all subjects in DICOM_DIR to 3D numpy arrays """
    subjects = os.listdir(dir_path)
    org, img, voxel_spacing, initial_patient_position = read_subject(subjects, dir_path) # d h w

    org = org[:,::-1,:,...].copy()
    img = img[:,::-1,:,...].copy()

    swapped_img = img.swapaxes(0, 2) # w h d

    dcm_w, dcm_h, dcm_d = swapped_img.shape

    img_tensor = resizing_windowing(swapped_img)
    
    return org, img_tensor, dcm_w, dcm_h, dcm_d, (voxel_spacing, initial_patient_position)