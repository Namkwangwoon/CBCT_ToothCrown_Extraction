from lib2to3.pgen2 import grammar
import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset_heatmap import get_val_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d_heatmap import UNet3D
from losses import DiceLoss
from transforms import (train_transform_cuda, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

import nibabel as nib
import numpy as np

import utils

def save_result(image, target, ground_truth, idx, save_dir='./results_heatmap_trans'):
    print('image.shape : ', image.shape)
    if len(image.shape) == 3:
        np_image = image.detach().cpu().numpy()
    else:
        np_image = image.squeeze(0).squeeze(0).detach().cpu().numpy()
    print('np_image.shape : ', np_image.shape)
    
    target= target.squeeze(0)
    np_target = torch.sum(target, axis=0)
    np_target= np_target.detach().cpu().numpy()

    kp_arr = np.zeros((64, 128, 128))
    gtkp_arr = np.zeros((64, 128, 128))
    # print('target ; ', target.shape)
    target2kp = utils.get_maximum_point(target.detach().cpu().numpy())
    target2kp = target2kp.numpy()
    target2kp = target2kp.astype(int)

    gt_points = utils.compute_3D_coordinate(np_image)
    for gtkp in gt_points:
        # gtkp_arr[gtkp[0]][gtkp[1]][gtkp[2]] = 1
        gtkp_arr[gtkp[0]][gtkp[1]][gtkp[2]] = 1

    for kp in target2kp:
        kp_arr[kp[0]][kp[1]][kp[2]] = 1

    # gtkp_arr = np.transpose(gtkp_arr, (2, 1, 0))
    # kp_arr = np.transpose(kp_arr, (2, 1, 0))

    nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nii_target = nib.Nifti1Image(np_target, affine=np.eye(4))
    nii_kp = nib.Nifti1Image(kp_arr, affine=np.eye(4))
    nii_gt_kp = nib.Nifti1Image(gtkp_arr, affine=np.eye(4))

    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(idx))
    nib.save(nii_target, save_dir + '/heatmap_{}.nii.gz'.format(idx))
    nib.save(nii_kp, save_dir + '/keypoints_{}.nii.gz'.format(idx))
    nib.save(nii_gt_kp, save_dir + '/gt_keypoints_{}.nii.gz'.format(idx))

model = UNet3D(in_channels=1 , num_classes= 4)

MODEL_WEIGHT_PATH = './checkpoints/epoch174_valLoss0.4417979419231415.pth'
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))

model = model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 

val_dataloader = get_val_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

model.eval()

for idx, data in enumerate(val_dataloader):
    if idx > 2:
        break
    image, ground_truth = data['image'], data['label']

    target = model(image)

    '''
    print('image : ', image.shape)
    print('ground_truth : ', ground_truth.shape)
    print('target : ', target.shape)

    image :  torch.Size([1, 1, 64, 128, 128])
    ground_truth :  torch.Size([1, 1, 4, 64, 128, 128])
    target :  torch.Size([1, 4, 64, 128, 128])
    '''

    # ground_truth = ground_truth.squeeze(0)
    save_result(image, target, ground_truth, idx)