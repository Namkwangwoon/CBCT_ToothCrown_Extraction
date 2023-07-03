from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    AddChanneld,
    DivisiblePadd,
    ThresholdIntensityd,
)

#Cuda version of "train_transform"
train_transform_cuda = Compose(
    [   
        # AddChanneld(keys=["image", "label"]),
        AddChanneld(keys=["image"]),
        # Spacingd(keys=['image', 'mask'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        ThresholdIntensityd(keys='image', threshold=1000, above=True, cval=0.0),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys=['image', 'mask'], factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        # DivisiblePadd(k=16, keys=["image", "label"]),
<<<<<<< HEAD
        ToTensord(keys=['image', 'cls', 'bbox', 'center', 'heatmap', 'mask'], device='cuda')
=======
        ToTensord(keys=['image', 'bbox', 'mask'], device='cuda')
>>>>>>> parent of de2f1a1... feat : add GD Loss
    ]
)

#Cuda version of "val_transform"
val_transform_cuda = Compose(
    [   
        AddChanneld(keys=["image"]),
        ThresholdIntensityd(keys='image', threshold=1000, above=True, cval=0.0),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        # DivisiblePadd(k=16, keys=["image", "label"]),
<<<<<<< HEAD
        ToTensord(keys=['image', 'cls', 'bbox', 'center', 'heatmap', 'mask'], device='cuda')
=======
        ToTensord(keys=['image', 'bbox', 'mask'], device='cuda')
>>>>>>> parent of de2f1a1... feat : add GD Loss
    ]
)