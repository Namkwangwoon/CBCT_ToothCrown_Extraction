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
# from unet3d_heatmap import UNet3D
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
def save_result(image, target, gt_cls, gt_heatmap, person_id, mask, bbox, idx=1, save_dir='./results'):
    print('image.shape : ', image.shape)
    if len(image.shape) == 3:
        np_image = image.detach().cpu().numpy()
    else:
        np_image = image.squeeze(0).squeeze(0).detach().cpu().numpy()
    print('np_image.shape : ', np_image.shape)
    
    target, coor1, coor2 = target
    cls = gt_cls.max(-1)[1].type(torch.int64)
    
    gt_cls = (torch.sum(gt_cls, dim=1)==1)[0]
    target = target[0, gt_cls]
    '''
    print("image : ", image.shape)
    print("target : ", target.shape)
    print("gt_heatmap : ", gt_heatmap.shape)
    image.shape :  torch.Size([1, 1, 128, 128, 128])
    np_image.shape :  (128, 128, 128)
    image :  torch.Size([1, 1, 128, 128, 128])
    target :  torch.Size([13, 64, 64, 64])
    gt_heatmap :  torch.Size([1, 13, 3])
    '''
    pred_hadamard_keypoints = utils.hadamard_product(target)
    pred_hadamard_keypoints = pred_hadamard_keypoints.detach().cpu().numpy()
    nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(idx))
    
    target = target.detach().cpu().numpy()
    gt_heatmap = gt_heatmap.squeeze(0).detach().cpu().numpy()
    pred_keypoints = utils.get_maximum_point(target)
    pred_keypoints = pred_keypoints.numpy()
    
    coor1 = utils._tranpose_and_gather_feature(coor1, cls)
    coor2 = utils._tranpose_and_gather_feature(coor2, cls)
    
    target_box = torch.cat([coor1, coor2], dim=-1).cuda()
    
    target_box_volumes = utils.draw_boxes(target_box)
    np_target_box_volumes = target_box_volumes.detach().cpu().numpy()
    
    gt_box_volumes = utils.draw_boxes(bbox)
    np_gt_box_volumes = gt_box_volumes.detach().cpu().numpy()
    
    for i in range(target.shape[0]):
        pred_heatmap = target[i]
        nii_pred_heatmap = nib.Nifti1Image(pred_heatmap, affine=np.eye(4))
        nib.save(nii_pred_heatmap, save_dir + '/pred_heatmap_{}.nii.gz'.format(i))
        
        
        # gtkp_arr = np.zeros((64, 64, 64))
        gtkp_arr = np.zeros((128, 128, 128))
        gtkp = gt_heatmap[i].astype(int)
        # gtkp = gtkp / 2
        gtkp = gtkp.astype(int)
        gtkp_arr[gtkp[0]][gtkp[1]][gtkp[2]] = 1
        nii_gtkp_arr = nib.Nifti1Image(gtkp_arr, affine=np.eye(4))
        nib.save(nii_gtkp_arr, save_dir + '/gt_point_{}.nii.gz'.format(i))
        
        
        # gtkp_arr = utils.resize_img(gtkp_arr, (128, 128, 128))
        # gtkp_arr = gtkp_arr.numpy()
        # nii_gtkp_arr = nib.Nifti1Image(gtkp_arr, affine=np.eye(4))
        # nib.save(nii_gtkp_arr, save_dir + '/128_gt_point_{}.nii.gz'.format(i))
        # predkp_arr = np.zeros((64, 64, 64))
        predkp_arr = np.zeros((128, 128, 128))
        predkp = pred_keypoints[i].astype(int)
        predkp_arr[predkp[0]][predkp[1]][predkp[2]] = 1
        nii_predkp_arr = nib.Nifti1Image(predkp_arr, affine=np.eye(4))
        nib.save(nii_predkp_arr, save_dir + '/pred_point_{}.nii.gz'.format(i))
        
        
        predhadamard_arr = np.zeros((128, 128, 128))
        predkp_hadamard = pred_hadamard_keypoints[i].astype(int)
        predhadamard_arr[predkp_hadamard[0]][predkp_hadamard[1]][predkp_hadamard[2]] = 1
        nii_predkp_hadamard = nib.Nifti1Image(predhadamard_arr, affine=np.eye(4))
        nib.save(nii_predkp_hadamard, save_dir + '/pred_hadamard_point_{}.nii.gz'.format(i))
        
        box_volume = np_target_box_volumes[i]
        nii_box = nib.Nifti1Image(box_volume, affine=np.eye(4))
        nib.save(nii_box, save_dir + '/pred_box_{}.nii.gz'.format(i))
        
        gt_box_volume = np_gt_box_volumes[i]
        nii_gt_box = nib.Nifti1Image(gt_box_volume, affine=np.eye(4))
        nib.save(nii_gt_box, save_dir + '/gt_box_{}.nii.gz'.format(i))
        

        
    '''
    # for gtkp in gt_points:
    #     gtkp_arr[gtkp[0]][gtkp[1]][gtkp[2]] = 1
    # for kp in target2kp:
    #     kp_arr[kp[0]][kp[1]][kp[2]] = 1
    '''
model = HeatmapVNet()

MODEL_WEIGHT_PATH = './checkpoints/epoch12_valLoss0.09383901953697205.pth'
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
model = model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda
val_dataloader = get_val_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)
model.eval()

for idx, data in enumerate(val_dataloader):
    image, gt_heatmap, gt_cls, person_id, mask, bbox = data['image'], data['center'], data['cls'], data['person_id'], data['mask'], data['bbox']
    # _, target = model(image)
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
    save_result(image, target, gt_cls, gt_heatmap, person_id, mask, bbox, idx)
    break