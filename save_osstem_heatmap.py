from lib2to3.pgen2 import grammar
import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
# from dataset_heatmap import get_val_Dataloaders
from dataset_osstem import get_val_Dataloaders
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
from unet3d_heatmap import UNet3D
# from losses import DiceLoss
# from transforms import (train_transform_cuda, val_transform_cuda)
from osstem_transforms import (train_transform_cuda, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

import nibabel as nib
import numpy as np

import utils


from vnet3d import HeatmapVNet
from tqdm import tqdm



def save_result(image, target, gt_cls, gt_heatmap, person_id, mask, save_dir='./results_heatmap_trans'):
    print('image.shape : ', image.shape)
    if len(image.shape) == 3:
        np_image = image.detach().cpu().numpy()
    else:
        np_image = image.squeeze(0).squeeze(0).detach().cpu().numpy()
    print('np_image.shape : ', np_image.shape)
    
    gt_cls = (torch.sum(gt_cls, dim=1)==1)[0]
    target = target[0, gt_cls]
    
    # print("target : ", target.shape)
    # print("values : ", target.max(1)[0].shape)
    # print("values : ", target.max(1)[0])
    # print("indices : ", target.max(1)[1].shape)
    # print("indices : ", target.max(1)[1])

    
    '''
    # np_target = torch.sum(target, axis=0)
    # np_target= np_target.detach().cpu().numpy()
    # kp_arr = np.zeros((64, 128, 128))
    # gtkp_arr = np.zeros((64, 128, 128))
    # print('target ; ', target.shape)
    # target2kp = utils.get_maximum_point(target.detach().cpu().numpy())
    # target2kp = target2kp.numpy()
    # target2kp = target2kp.astype(int)

    # gt_points = utils.compute_3D_coordinate(np_image)
    # for gtkp in gt_points:
    #     # gtkp_arr[gtkp[0]][gtkp[1]][gtkp[2]] = 1
    #     gtkp_arr[gtkp[0]][gtkp[1]][gtkp[2]] = 1

    # for kp in target2kp:
    #     kp_arr[kp[0]][kp[1]][kp[2]] = 1

    # gtkp_arr = np.transpose(gtkp_arr, (2, 1, 0))
    # kp_arr = np.transpose(kp_arr, (2, 1, 0))
    '''
    
    target_idx = torch.round(utils.hadamard_product(target)).int()
    target_map = torch.zeros_like(target)
    
    for i, t in zip(target_idx, target_map):
        t[i[0]][i[1]][i[2]] += 1
    
    
    
    # np_target= target.detach().cpu().numpy()
    np_target = target_map.detach().cpu().numpy()
    np_gt_heatmap = gt_heatmap.squeeze(0).detach().cpu().numpy()
    
    # print()
    # print("np_image :", np_image.shape)
    # print("np_target :", np_target.shape)
    # print("np_gt_heatmap :", np_gt_heatmap.shape)
    
    person_id = person_id[0]

    nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(person_id))
    
    # print("mask :", mask.shape)
    mask = mask.squeeze(0).detach().cpu().numpy()
    
    for i, m in enumerate(tqdm(mask)):
        nii_mask = nib.Nifti1Image(m, affine=np.eye(4))
        
        nib.save(nii_mask, save_dir + '/mask_{}_{}.nii.gz'.format(person_id, i))
    
    # for i, (t, g) in enumerate(zip(tqdm(np_target), np_gt_heatmap)):
        
    #     # nii_target = nib.Nifti1Image(t, affine=np.eye(4))
    #     nii_gt = nib.Nifti1Image(g, affine=np.eye(4))
        
    #     # nib.save(nii_target, save_dir + '/target_heatmap_{}_{}.nii.gz'.format(person_id, i))
    #     # nib.save(nii_target, save_dir + '/target_point_{}_{}.nii.gz'.format(person_id, i))
    #     nib.save(nii_gt, save_dir + '/gt_{}_{}.nii.gz'.format(person_id, i))
        
        

# model = UNet3D(in_channels=1 , num_classes= 4)
model = HeatmapVNet()


MODEL_WEIGHT_PATH = './checkpoints/epoch35_valLoss3.63382319790162e-08.pth'
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))

model = model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 

val_dataloader = get_val_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

model.eval()

for idx, data in enumerate(val_dataloader):
    image, gt_heatmap, gt_cls, person_id, mask = data['image'], data['bbox'], data['cls'], data['person_id'], data['mask']

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
    save_result(image, target, gt_cls, gt_heatmap, person_id, mask)
    
    break
    