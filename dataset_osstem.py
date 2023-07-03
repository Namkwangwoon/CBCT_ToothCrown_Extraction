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
from utils import resize_img, generate_gaussian_heatmap
from tqdm import tqdm


TRAIN = ['6', '7', '8', '9', '12', '13', '15', '19', '23', '25', '26', '28', '29', '30', '31', '32',
         '33', '34', '35', '37', '38', '39', '41', '44', '46', '47', '48', '49', '50', '51', '52']
# 21, 36
VAL = ['1', '2', '3', '4', '5', '10', '11', '14', '16', '17']
TEST = ['18', '20', '22', '24', '27', '40', '42', '45']
# TEST = [ '1', '2', '3', '4', '5', '10', '11', '14', '16', '17', '6', '7', '8', '9', '12', '13', '15', '19', '21', '25', '26', '28', '29', '30', '31', '32', '33',
#          '34', '35', '36', '37', '38', '39', '41', '44', '46', '47', '48', '49', '50', '51', '52']

# TOOTH_NUM = ['11', '12', '13', '14', '15', '16', '17', '18',
#              '21', '22', '23', '24', '25', '26', '27', '28',
#              '31', '32', '33', '34', '35', '36', '37', '38',
#              '41', '42', '43', '44', '45', '46', '47', '48']

UPPER_TOOTH_NUM = ['11', '12', '13', '14', '15', '16', '17', '18', 
            '21', '22', '23', '24', '25', '26', '27', '28']
LOWER_TOOTH_NUM = ['41', '42', '43', '44', '45', '46', '47', '48',
                '31', '32', '33', '34', '35', '36', '37', '38',]

OUTLIER = {'1' : ['34'],
           '2' : ['12', '13'],
           '3' : ['17', '34', '36', '42', '43', '44', '45', '47'],
           '4' : ['16', '23', '25', '35', '37', '42', '47'],
           '5' : ['35'],
           '10' : ['11'],
           '11' : ['11'],
           '14' : ['31'],
           '16' : ['42'],
           '17' : ['43'],
           '18' : ['13'],
           '22' : ['33'],
           '24' : ['13'],
           '27' : ['45'],
           '40' : ['16'],
           '42' : ['46'],
           '45' : ['25']}


'''
43번 환자의 Mask 데이터는 아예 존재하지 않음..
'''


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
        # self.dir = "../datasets/osstem_clean"
        # self.dir = "F:/osstem_clean"
        # '''
        # C:\Users\JeeheonKim\source\ct\pose\realigned
        # '''
        # self.dir = os.path.abspath("C:/Users/JeeheonKim/source/ct/pose/realigned")
        self.dir = os.path.abspath("F:/osstem_clean2")
        self.meta = dict()
        self.transform = transforms

        self.mode = mode
        if self.mode == 'train':
            self.data = TRAIN + VAL
        else:
            self.data = TEST
            
        self.input = '1.nii.gz'
        # self.input = 'crop_image.nii.gz'
        self.box_anno = 'bbox.json'
        # self.box_anno = 'lower_bbox.json'
        self.whole_mask = 'whole_mask.nii.gz'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        person_id = self.data[idx]
        print("< person", person_id, ">")
        
        file_list = sorted(os.listdir(os.path.join(self.dir, person_id, 'nii')))
        img_path = os.path.join(self.dir, person_id, 'nii', self.input)
        box_anno_path = os.path.join(self.dir, person_id, 'nii', self.box_anno)
        mask_list = set(file_list) - set([self.input, self.box_anno, self.whole_mask])
        if person_id in OUTLIER.keys():
            mask_list = mask_list - set(OUTLIER[person_id])
        mask_list = list(mask_list)
        
        img_object = nib.load(img_path)
        img_array = img_object.get_fdata()
        img_array = img_array.astype(int)
        img_array = torch.Tensor(img_array).permute(-1, 1, 0)
        d, h, w = img_array.shape
        img_array = torch.flip(img_array, [0, 1])
        # print("img :", img_array.shape)
        
        # image = nib.Nifti1Image(np.array(img_array), affine=np.eye(4))
        # nib.save(image, 'image_{}.nii.gz'.format(person_id))
        
        with open(box_anno_path, 'r') as file:
            bbox_anno = json.load(file)
            
        # Remove outlier teeth
        annos = list(bbox_anno.keys())
        if person_id in OUTLIER.keys():
            print("teeth")
            print(annos)
            annos = sorted(list(set(annos) - set(OUTLIER[person_id])))
            print(">>>", annos)
        
        cls, bbox, heatmaps, mask = [], [], [], []
        for tooth in tqdm(annos, desc='(Annotating...) ', ascii=' ='):
            cls.append(np.expand_dims((np.array(TOOTH_NUM)==tooth)*1, axis=0))
            
            box = bbox_anno[tooth]
            # bbox -> heatmap
            box = np.array([(box[0]+box[3])/2, (box[1]+box[4])/2, (box[2]+box[5])/2]) / np.array(img_array.shape) * np.array([128, 128, 128])
            bbox.append(np.expand_dims(box, axis=0))
            heatmap = generate_gaussian_heatmap((128, 128, 128), box)
            heatmaps.append(np.expand_dims(heatmap, axis=0))
            
            mask_object = nib.load(os.path.join(self.dir, person_id, 'nii', tooth+"_gt.nii.gz"))
            mask_array = mask_object.get_fdata()
            mask_array = resize_img(mask_array, (128, 128, 128))
            mask.append(np.expand_dims(mask_array, axis=0))
        
        img_array = resize_img(np.array(img_array), (128, 128, 128))
        np_cls = np.concatenate(cls, axis=0)
        np_bbox = np.concatenate(bbox, axis=0)
        np_heatmap = np.concatenate(heatmaps, axis=0)
        np_mask = np.concatenate(mask, axis=0)
        
        proccessed_out = {
            'person_id': person_id,
            'image': img_array,
            'cls': np_cls,
            'bbox': np_bbox,
            'heatmap': np_heatmap,
            'mask' : np_mask
        }
        
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out


class RealignedCT(Dataset):
    """
    The base dataset class for Decathlon segmentation tasks
    -- __init__()
    :param task_number -> represent the organ dataset ID (see task_names above for hints)
    :param dir_path -> the dataset directory path to .tar files
    :param transform -> optional - transforms to be applied on each instance
    """
    def __init__(self, transforms = None, mode = None) -> None:
        super(RealignedCT, self).__init__()
        # self.dir = "../datasets/osstem_clean"
        # self.dir = "F:/osstem_clean"
        # '''
        # C:\Users\JeeheonKim\source\ct\pose\realigned
        # '''
        self.dir = os.path.abspath("../datasets/realigned")
        self.meta = dict()
        self.transform = transforms

        self.mode = mode
        if self.mode == 'train':
            self.data = TRAIN + VAL
        else:
            self.data = TEST

        self.crop_data = []
        for single_data in self.data:
            self.crop_data.append((single_data, 'upper'))
            self.crop_data.append((single_data, 'lower'))
            
        # self.input = '1.nii.gz'
        # self.input = 'crop_image.nii.gz'
        # self.box_anno = 'bbox.json'
        self.box_anno = 'lower_bbox.json'
        self.whole_mask = 'whole_mask.nii.gz'

    def __len__(self):
        return len(self.crop_data)

    def __getitem__(self, idx):
        person_id, flag = self.crop_data[idx]
        # print('Person : ', person_id, ' Flag : ', flag)

        if flag == 'upper':
            crop_image_name = 'crop_image_upper.nii.gz'
            self.box_anno = 'upper_bbox.json'
            TOOTH_NUM = UPPER_TOOTH_NUM
        elif flag == 'lower':
            crop_image_name = 'crop_image.nii.gz'
            self.box_anno = 'lower_bbox.json'
            TOOTH_NUM = LOWER_TOOTH_NUM

        file_list = os.listdir(os.path.join(self.dir, person_id))
        new_file_list = []
        for file_name in file_list:
            if file_name[3:8] == flag:
                new_file_list.append(file_name)
        file_list = sorted(new_file_list)

        # img_path = os.path.join(self.dir, person_id, 'nii', self.input)
        img_path = os.path.join(self.dir, person_id, crop_image_name)
        box_anno_path = os.path.join(self.dir, person_id, self.box_anno)

        # mask_list = set(file_list) - set([self.input, self.box_anno, self.whole_mask])
        mask_list = set(file_list)
        if person_id in OUTLIER.keys():
            mask_list = mask_list - set(OUTLIER[person_id])
        mask_list = list(mask_list)
        
        img_object = nib.load(img_path)
        img_array = img_object.get_fdata()
        # img_array = img_array.astype(int)
        # img_array = torch.Tensor(img_array).permute(-1, 1, 0)
        d, h, w = img_array.shape
        # img_array = torch.flip(img_array, [0, 1])
        # print("img :", img_array.shape)
        
        # image = nib.Nifti1Image(np.array(img_array), affine=np.eye(4))
        # nib.save(image, 'image_{}.nii.gz'.format(person_id))
        
        with open(box_anno_path, 'r') as file:
            bbox_anno = json.load(file)
            
        # Remove outlier teeth
        annos = list(bbox_anno.keys())
        if person_id in OUTLIER.keys():
            # print("teeth")
            # print(annos)
            annos = sorted(list(set(annos) - set(OUTLIER[person_id])))
            # print(">>>", annos)
        
        cls, bboxes, centers, heatmaps, mask = [], [], [], [], []
        # for tooth in tqdm(annos, desc='(Annotating...) ', ascii=' ='):
        for tooth in annos:
            cls.append(np.expand_dims((np.array(TOOTH_NUM)==tooth)*1, axis=0))
            
            box = bbox_anno[tooth]
            
            bbox = np.array(box) / np.concatenate([img_array.shape, img_array.shape])
            bboxes.append(np.expand_dims(bbox, axis=0))
            
            # bbox -> heatmap
            # box = np.array([(box[0]+box[3])/2, (box[1]+box[4])/2, (box[2]+box[5])/2]) / np.array(img_array.shape) * np.array([64, 64, 64])
            center = np.array([(box[0]+box[3])/2, (box[1]+box[4])/2, (box[2]+box[5])/2]) / np.array(img_array.shape) * np.array([128, 128, 128])

            # box = box / np.array(img_array.shape) * np.array([64, 64, 64])

            centers.append(np.expand_dims(center, axis=0))
            # heatmap = generate_gaussian_heatmap((128, 128, 128), box)
            heatmap = generate_gaussian_heatmap((128, 128, 128), center)
            heatmaps.append(np.expand_dims(heatmap, axis=0))
            
            # mask_object = nib.load(os.path.join(self.dir, person_id, 'nii', tooth+"_gt.nii.gz"))
            mask_object = nib.load(os.path.join(self.dir, person_id, tooth+"_{}_crop.nii.gz".format(flag)))
            mask_array = mask_object.get_fdata()
            mask_array = resize_img(mask_array, (128, 128, 128))
            mask.append(np.expand_dims(mask_array, axis=0))
        
        # img_array = resize_img(np.array(img_array), (128, 128, 128))
        # img_array = resize_img(img_array, (128, 128, 128))
        img_array = resize_img(img_array, (128, 128, 128))
        np_cls = np.concatenate(cls, axis=0)
        np_bbox = np.concatenate(bboxes, axis=0)
        np_center = np.concatenate(centers, axis=0)
        np_heatmap = np.concatenate(heatmaps, axis=0)
        np_mask = np.concatenate(mask, axis=0)
        
        '''
        영상 : [1, 25, 32]

        상악 : [1, 25, 32 or 16] : 
        하악 : [1, 25, 32 or 16] : [0, 0, 0, 0, ] + [1, 0, 0 ,0 ,0 ]


        '''
        proccessed_out = {
            'person_id': person_id,
            'image': img_array,
            'cls': np_cls,
            'bbox': np_bbox,
            'center': np_center,
            'heatmap': np_heatmap,
            'mask' : np_mask
        }
        
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out


def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    # dataset = MedicalSegmentationDecathlon(transforms=[train_transforms, val_transforms, test_transforms])
    train_dataset = RealignedCT(transforms=train_transforms, mode='train')
    val_dataset = RealignedCT(transforms=val_transforms, mode='val')

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
    val_dataset = RealignedCT(transforms=val_transforms, mode='val')

    #Spliting dataset and building their respective DataLoaders
    # val_set = copy.deepcopy(dataset)
    # val_set.set_mode('val')
    val_dataloader = DataLoader(dataset= val_dataset, batch_size= VAL_BATCH_SIZE, shuffle= False)
    
    return val_dataloader