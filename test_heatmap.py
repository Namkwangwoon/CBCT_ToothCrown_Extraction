from lib2to3.pgen2 import grammar
import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset import get_val_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d_heatmap import UNet3D
from losses import DiceLoss
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

import nibabel as nib
import numpy as np

import utils

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def save_result(image, target, ground_truth, idx, save_dir='./results_heatmap_trans'):
    print('>> bef << ')
    print('image.shape : ', image.shape)
    print('target.shape : ', target.shape)
    print('ground_truth.shape : ', ground_truth.shape)
    print()

    image = image.squeeze(0).squeeze(0)
    # target = target.squeeze(0).squeeze(0)
    target = target.squeeze(0)
    # ground_truth = ground_truth.squeeze(0).squeeze(0)


    np_gt = ground_truth.squeeze(0).squeeze(0).detach().cpu().numpy()

    # target_pts, gt_pts = utils.compute_MIP_and_coordinates(np_target, np_gt)
    gt_pts = utils.compute_3D_coordinate(np_gt)
    # print('gt_pts : ', gt_pts)
    gt_heatmaps = utils.kp2heatmap(gt_pts, size=(64, 128, 128))
    print('gt_heatmaps : ', gt_heatmaps.shape)

    np_gt = gt_heatmaps.numpy()
    np_image = image.detach().cpu().numpy()
    np_target = target.detach().cpu().numpy()
    for ii in range(4):
        single_target = np_target[ii, :, :, :]
        # single_target = single_target.transpose(2, 1, 0)

        single_heatmap = nib.Nifti1Image(single_target, affine=np.eye(4))
        nib.save(single_heatmap, save_dir + '/single_heatmap_{}_{}.nii.gz'.format(ii, idx))


        single_gt = np_gt[ii, :, :, :]
        # single_gt = single_gt.transpose(2, 1, 0)

        single_gt_heatmap = nib.Nifti1Image(single_gt, affine=np.eye(4))
        nib.save(single_gt_heatmap, save_dir + '/single_gt_heatmap_{}_{}.nii.gz'.format(ii, idx))

    # np_target = softmax(np_target)
    # np_target = nn.Softmax()  
    # np_gt = ground_truth.detach().cpu().numpy()

    target2kp = utils.get_maximum_point(np_target)
    target2kp = target2kp.numpy()
    target2kp = target2kp.astype(int)

    # print('target2kp : ', target2kp.shape)
    kp_arr = np.zeros((image.shape))
    # gt_kp_arr = np.zeros((image.shape))

    print('target2kp : ', target2kp)
    for kp in target2kp:
        # print(kp)
        kp_arr[kp[0]][kp[1]][kp[2]] = 1

    # for gt_kp in gt_pts:
    #     # print(kp)
    #     gt_kp_arr[gt_kp[0]][gt_kp[1]][gt_kp[2]] = 1


    '''
    print('np_image : ', np_image.shape)
    print('np_target : ', np_target.shape)
    print('np_gt : ', np_gt.shape)

    np_image :  (64, 128, 128)
    np_target :  (4, 64, 128, 128)
    np_gt :  (64, 128, 128)
    '''

    # np_target = np_target.max()
    # np_target = np.max(np_target, axis=0)


    # nib.save(nii_target, save_dir + '/heatmap_{}.nii.gz'.format(idx))
    # nib.save(nii_target, save_dir + '/heatmap_{}.nii.gz'.format(idx))
    # nib.save(nii_target, save_dir + '/heatmap_{}.nii.gz'.format(idx))
    # nib.save(nii_target, save_dir + '/heatmap_{}.nii.gz'.format(idx))

    # image = np.swapaxes(image,0,2)
    # target = np.swapaxes(target,0,2)
    # ground_truth = np.swapaxes(ground_truth,0,2)

    # target = target.transpose(2, 1, 0)
    # ground_truth = ground_truth.transpose(2, 1, 0)

    # print('>> aft << ')
    # print('image.shape : ', image.shape)
    # print('target.shape : ', target.shape)
    # print('ground_truth.shape : ', ground_truth.shape)
    # print()

    nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nii_target = nib.Nifti1Image(np_target, affine=np.eye(4))
    nii_kp = nib.Nifti1Image(kp_arr, affine=np.eye(4))
    # nii_gt_kp = nib.Nifti1Image(gt_kp_arr, affine=np.eye(4))
    nii_gt = nib.Nifti1Image(ground_truth.squeeze(0).squeeze(0).detach().cpu().numpy(), affine=np.eye(4))

    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(idx))
    nib.save(nii_target, save_dir + '/heatmap_{}.nii.gz'.format(idx))
    nib.save(nii_kp, save_dir + '/keypoints_{}.nii.gz'.format(idx))
    # nib.save(nii_gt_kp, save_dir + '/gt_keypoints_{}.nii.gz'.format(idx))
    nib.save(nii_gt, save_dir + '/gt_alpha_{}.nii.gz'.format(idx))


model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)

# MODEL_WEIGHT_PATH = './checkpoints#1/epoch4_valLoss0.7130498807546755.pth'
# MODEL_WEIGHT_PATH = './checkpoints(voi detection)/epoch8_valLoss0.5606752634048462.pth'
# MODEL_WEIGHT_PATH = './checkpoints/epoch99_valLoss0.0004375985299702734.pth'
# MODEL_WEIGHT_PATH = './checkpoints/epoch5_valLoss0.003529988694936037.pth'
MODEL_WEIGHT_PATH = './checkpoints/epoch9_valLoss0.0024340334348380566.pth'

# model = torch.load(MODEL_WEIGHT_PATH)
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
# print(model)

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    # model = model.to('cuda')
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


val_dataloader = get_val_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

# criterion = nn.L1Loss()
criterion = nn.MSELoss()

min_valid_loss = math.inf
valid_losses = 0.0
model.eval()

for idx, data in enumerate(val_dataloader):
    if idx > 0:
        break
    image, ground_truth = data['image'], data['label']
    # image = image.float()
    # ground_truth = ground_truth.float()

    # print('image.shape : ', image.shape)
    # print('ground_truth.shape : ', ground_truth.shape)
    # print()

    target = model(image)

    save_result(image, target, ground_truth, idx)
    # print('image info : ', torch.max(image), ', ', torch.min(image))
    # print('target info : ', torch.max(target), ', ', torch.min(target))
    # print('ground_truth info : ', torch.max(ground_truth), ', ', torch.min(ground_truth))

    loss = criterion(target,ground_truth)
    valid_losses += loss.item()

    # print('Val DICE Loss : ', loss)
    print('Val MSE LOSS : ', loss.item())
    # print('Val DICE Loss : ', valid_loss)
    
print()
print(f'=> Total Validation Score: {valid_losses / len(val_dataloader)}')