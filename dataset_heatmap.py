import copy
import nibabel as nib
import numpy as np
import os
import tarfile
import json
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from skimage.transform import resize
from config import (
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)

import utils
from utils import resize_img

class MedicalSegmentationDecathlon(Dataset):
    """
    The base dataset class for Decathlon segmentation tasks
    -- __init__()
    :param task_number -> represent the organ dataset ID (see task_names above for hints)
    :param dir_path -> the dataset directory path to .tar files
    :param transform -> optional - transforms to be applied on each instance
    """
    def __init__(self, transforms = None, mode = None) -> None:
        super(MedicalSegmentationDecathlon, self).__init__()
        self.dir = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/ban_nii')
        self.meta = dict()
        self.transform = transforms

        self.mode = mode
        self.data = sorted(os.listdir(os.path.join(self.dir, mode)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data[idx]

        img_path = os.path.join(self.dir, self.mode, name, 'data.nii.gz')
        label_path = os.path.join(self.dir, self.mode, name, 'gt_alpha.nii.gz')

        img_object = nib.load(img_path)
        label_object = nib.load(label_path)

        img_array = img_object.get_fdata()
        label_array = label_object.get_fdata()

        img_array = resize_img(img_array, (64, 128, 128))
        label_array = resize_img(label_array, (64, 128, 128))
        numpy_label = label_array.numpy()

        list_label_points = utils.compute_3D_coordinate(numpy_label)
        tensor_heatmaps = utils.kp2heatmap(list_label_points, size=(64, 128, 128))

        # print('tensor_heatmaps : ', tensor_heatmaps.shape)

        # img_array = resize(img_array, (128, 256, 256))
        # label_array = resize(label_array, (128, 256, 256))

        # label_array = np.moveaxis(label_array, -1, 0)
        # proccessed_out = {'name': name, 'image': img_array, 'label': label_array}
        proccessed_out = {
            'name': name, 
            'image': img_array, 
            'label': tensor_heatmaps,
            'label_array' : label_array
        }
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out



def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    # dataset = MedicalSegmentationDecathlon(transforms=[train_transforms, val_transforms, test_transforms])
    train_dataset = MedicalSegmentationDecathlon(transforms=train_transforms, mode='train')
    val_dataset = MedicalSegmentationDecathlon(transforms=val_transforms, mode='val')

    train_dataloader = DataLoader(dataset= train_dataset, batch_size= TRAIN_BATCH_SIZE, shuffle= False)
    val_dataloader = DataLoader(dataset= val_dataset, batch_size= VAL_BATCH_SIZE, shuffle= False)
    # test_dataloader = DataLoader(dataset= test_set, batch_size= TEST_BATCH_SIZE, shuffle= False)
    test_dataloader = None
    
    return train_dataloader, val_dataloader, test_dataloader

def get_val_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    # dataset = MedicalSegmentationDecathlon(transforms=[train_transforms, val_transforms, test_transforms])
    val_dataset = MedicalSegmentationDecathlon(transforms=val_transforms, mode='val')

    #Spliting dataset and building their respective DataLoaders
    # val_set = copy.deepcopy(dataset)
    # val_set.set_mode('val')
    val_dataloader = DataLoader(dataset= val_dataset, batch_size= VAL_BATCH_SIZE, shuffle= False)
    
    return val_dataloader