# from unet3d_heatmap import UNet3D
from vnet3d import HeatmapVNet

from osstem_transforms import (train_transform_cuda, val_transform_cuda)
from dataset_osstem import get_train_val_test_Dataloaders
from losses import BinaryFocalLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

model = HeatmapVNet()

model = model.cuda()


train_transforms = train_transform_cuda
val_transforms = val_transform_cuda

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transforms, val_transforms=val_transforms,
                                                                  test_transforms=val_transforms)


# criterion2 = nn.MSELoss()
criterion2 = BinaryFocalLoss()
# criterion3 = nn.L1Loss()

optimizer = Adam(params=model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [40, 120, 360], gamma=0.1, last_epoch=-1)
# get_last_lr


for idx, data in enumerate(train_dataloader):
    print()
    print("Epoch[%d]" % (idx+1))
    print()