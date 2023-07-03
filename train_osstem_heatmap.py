import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset_osstem import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
# from unet3d_heatmap import UNet3D
# from vnet3d import HeatmapVNet, HeatmapVNetx3
from vnet3d import HeatmapVNet

from osstem_transforms import (train_transform_cuda, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

<<<<<<< HEAD
from losses import BinaryFocalLoss, GDLoss
from utils import hadamard_product, _tranpose_and_gather_feature
=======
from losses import BinaryFocalLoss
import utils
>>>>>>> parent of de2f1a1... feat : add GD Loss

# from test_heatmap import save_result

import nibabel as nib
import numpy as np

import gc
import torch
from torch.optim.lr_scheduler import MultiStepLR

gc.collect()
torch.cuda.empty_cache()

writer = SummaryWriter()

# def _log_params(model, writer, num_iterations):
#     for name, value in model.named_parameters():
#         if value.grad is not None :
#             writer.add_histogram(name, value.data.cpu().numpy(), num_iterations)
#             writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), num_iterations)
#         else :
#             print(name, value.grad)

model = HeatmapVNet()
## model = HeatmapVNetx3()

model = model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

# criterion = DiceLoss()
# criterion2 = nn.MSELoss()
<<<<<<< HEAD
criterion1 = BinaryFocalLoss()
criterion2 = nn.L1Loss()
gd_criterion = GDLoss()
box_criterion = nn.MSELoss()
=======
criterion = BinaryFocalLoss()
>>>>>>> parent of de2f1a1... feat : add GD Loss

optimizer = Adam(params=model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [40, 120, 360], gamma=0.1, last_epoch=-1)
# get_last_lr

# from save_heatmap import save_result

min_valid_loss = math.inf
# alpha_pts = 100
# alpha_heatmap = 100

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    model.train()

    # print('optimizer.param_groups : ', scheduler.get_last_lr())
    print()
    print('EPOCH : {} / {}, LR : {}, len(train_dataloader) : , '.format(epoch, TRAINING_EPOCH, optimizer.param_groups[0]["lr"]), len(train_dataloader))
    for idx, data in enumerate(train_dataloader):
        # if idx > 0:
        #     break
<<<<<<< HEAD
        image, bbox, center, heatmap, cls = data['image'], data['bbox'].type(torch.float32), data['center'], data['heatmap'], data['cls']
        
        # target, box = model(image)
        target, coor1, coor2 = model(image)
        target = target[0, (torch.sum(cls, dim=1)==1)[0], :, :, :]
        
        # print()
        # print("box :", box.shape)
        # print()
        
=======
        image, ground_truth, cls = data['image'], data['bbox'], data['cls']
        
        target = model(image)
        target = target[:, (torch.sum(cls, dim=1)==1)[0], :, :, :]
>>>>>>> parent of de2f1a1... feat : add GD Loss
        
        # print()
        # print("image :", image.shape)
        # print("gt :", ground_truth.shape)
        # print("cls :", torch.sum(cls, dim=1))
        # print("target :", target.shape)

        ## gt x3번 넣어줘야 함!
<<<<<<< HEAD
        train_loss_heatmap = criterion1(target, heatmap)
        
        train_target_idx = hadamard_product(target)
        
        train_loss_point = criterion2(train_target_idx / 128, center[0] / 128)
        
        train_gd_loss = gd_criterion(target, cls)
        
        cls = cls.max(-1)[1].type(torch.int64)
        coor1 = _tranpose_and_gather_feature(coor1, cls)
        coor2 = _tranpose_and_gather_feature(coor2, cls)
        
        target_box = torch.cat([coor1, coor2], dim=-1).cuda()
        
        train_loss_box = box_criterion(target_box, bbox)

        train_losses = train_loss_heatmap + train_loss_point + train_gd_loss + train_loss_box
        
        print(' {} / {} => Total loss : {} | Heatmap loss : {} | Point loss : {} | GD loss : {} | Box loss : {}'.format(
            idx+1, len(train_dataloader), train_losses.item(), train_loss_heatmap.item(), train_loss_point.item(), train_gd_loss.item(), train_loss_box.item())
        )
=======
        loss_heatmap = criterion(target, ground_truth)

        losses = loss_heatmap
        print(' {} / {} => Total loss : {}'.format(idx+1, len(train_dataloader), losses.item()))
>>>>>>> parent of de2f1a1... feat : add GD Loss

        losses.backward()
        # _log_params(model, writer, 1)
        optimizer.step()
        optimizer.zero_grad()

        train_loss += losses.item()
    
    scheduler.step()

    valid_losses = 0.0
    model.eval()
            
    print()
    print("Validation start !!")
    print()
            
    with torch.no_grad():
        for data in val_dataloader:
            
            
<<<<<<< HEAD
            image, bbox, center, heatmap, cls = data['image'], data['bbox'].type(torch.float32), data['center'], data['heatmap'], data['cls']
            target, coor1, coor2 = model(image)
            target = target[0, (torch.sum(cls, dim=1)==1)[0], :, :, :]
            
            
            ## gt 3번 넣어줘야 함!
            val_loss_heatmap = criterion1(target, heatmap)
            
            val_target_idx = hadamard_product(target)
        
            val_loss_point = criterion2(val_target_idx / 128, center[0] / 128)
            
            val_gd_loss = gd_criterion(target, cls)
            
            cls = cls.max(-1)[1].type(torch.int64)
            coor1 = _tranpose_and_gather_feature(coor1, cls)
            coor2 = _tranpose_and_gather_feature(coor2, cls)
            
            target_box = torch.cat([coor1, coor2], dim=-1)
            
            val_loss_box = box_criterion(target_box, bbox)
        
=======
            image, ground_truth, cls = data['image'], data['bbox'], data['cls']
            target = model(image)
            target = target[:, (torch.sum(cls, dim=1)==1)[0], :, :, :]
            
            
            ## gt 3번 넣어줘야 함!
            loss_heatmap = criterion(target, ground_truth)
>>>>>>> parent of de2f1a1... feat : add GD Loss
        
            # gt_points = utils.compute_3D_coordinate(image.squeeze(0).squeeze(0).detach().cpu().numpy())
            # pred_points = utils.get_maximum_point(target.squeeze(0).detach().cpu().numpy())

            # tensor_gt_points = torch.tensor(gt_points).cuda().float()
            # pred_points = pred_points.cuda()

            # loss_pts = criterion3(pred_points, tensor_gt_points)

            # valid_loss = (loss_pts / alpha_pts) + (alpha_heatmap * loss_heatmap)
<<<<<<< HEAD
            valid_loss = val_loss_heatmap + val_loss_point + val_gd_loss + val_loss_box
            
            print(' {} / {} => Total loss : {} | Heatmap loss : {} | Point loss : {} | GD loss : {} | Box loss : {}'.format(
                idx+1, len(val_dataloader), valid_loss.item(), val_loss_heatmap.item(), val_loss_point.item(), val_gd_loss.item(), val_loss_box.item()))
=======
            valid_loss = loss_heatmap
>>>>>>> parent of de2f1a1... feat : add GD Loss
            
            # print(' {} / {} => Point loss : {} | Heatmap loss : {} | Total loss : {}'.format(idx+1, len(train_dataloader), loss_pts.item() / 10, loss_heatmap.item() * 10, losses.item()))


            # valid_loss = 10 * valid_loss

            # valid_loss = (loss_pts / 10) + (loss_heatmap * 10)
            # valid_loss = loss_heatmap

            # print('Val Point Loss : {}, Heatmap Loss : {}'.format((loss_pts / 10), (loss_heatmap * 10)))
            valid_losses += valid_loss

            
        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_losses / len(val_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(val_dataloader)}')
        
        valid_losses = valid_losses / len(val_dataloader)

        if min_valid_loss > valid_losses:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
            min_valid_loss = valid_losses
            # Saving State Dict
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')
            
        print()

writer.flush()
writer.close()

