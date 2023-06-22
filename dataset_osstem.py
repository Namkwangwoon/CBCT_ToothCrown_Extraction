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


TRAIN = ['6', '7', '8', '9', '12', '13', '15', '19', '21', '25', '26', '28', '29', '30', '31', '32', '33',
         '34', '35', '36', '37', '38', '39', '41', '44', '46', '47', '48', '49', '50', '51', '52']
VAL = ['1', '2', '3', '4', '5', '10', '11', '14', '16', '17']
TEST = ['18', '20', '22', '24', '27', '40', '42', '45']
# TEST = [ '1', '2', '3', '4', '5', '10', '11', '14', '16', '17', '6', '7', '8', '9', '12', '13', '15', '19', '21', '25', '26', '28', '29', '30', '31', '32', '33',
#          '34', '35', '36', '37', '38', '39', '41', '44', '46', '47', '48', '49', '50', '51', '52']

TOOTH_NUM = ['11', '12', '13', '14', '15', '16', '17', '18',
             '21', '22', '23', '24', '25', '26', '27', '28',
             '31', '32', '33', '34', '35', '36', '37', '38',
             '41', '42', '43', '44', '45', '46', '47', '48']

OUTLIER = {'1' : ['22', '34'],
           '2' : ['12', '13'],
           '3' : ['17', '34', '36', '42', '43', '44', '45', '47'],
           '4' : ['16', '23', '25', '35', '37', '42', '47'],
           '5' : ['35'],
           '10' : ['11'],
           '11' : ['11'],
           '14' : ['22', '31'],
           '16' : ['42'],
           '17' : ['34', '43', '44'],
           '18' : ['13', '33'],
           '20' : ['32', '42'],
           '22' : ['33'],
           '23' : ['33'],
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
        self.dir = "../datasets/osstem_clean"
        self.meta = dict()
        self.transform = transforms

        self.mode = mode
        if self.mode == 'train':
            self.data = TRAIN + VAL
        else:
            self.data = TEST
            
        self.input = '1.nii.gz'
        self.box_anno = 'bbox.json'
        self.whole_mask = 'whole_mask.nii.gz'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # name = self.data[idx]

        # img_path = os.path.join(self.dir, self.mode, name, 'data.nii.gz')
        # label_path = os.path.join(self.dir, self.mode, name, 'gt_alpha.nii.gz')

        # img_object = nib.load(img_path)
        # label_object = nib.load(label_path)

        # img_array = img_object.get_fdata()
        # label_array = label_object.get_fdata()

        # img_array = resize_img(img_array, (64, 128, 128))
        # label_array = resize_img(label_array, (64, 128, 128))
        # numpy_label = label_array.numpy()

        # list_label_points = utils.compute_3D_coordinate(numpy_label)
        # tensor_heatmaps = utils.kp2heatmap(list_label_points, size=(64, 128, 128))

        # print('tensor_heatmaps : ', tensor_heatmaps.shape)

        # img_array = resize(img_array, (128, 256, 256))
        # label_array = resize(label_array, (128, 256, 256))

        # label_array = np.moveaxis(label_array, -1, 0)
        # proccessed_out = {'name': name, 'image': img_array, 'label': label_array}
        
        
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
        
        # kkk = nib.Nifti1Image(np.array(img_array), affine=np.eye(4))
        # nib.save(kkk, person_id+"kkk.nii.gz")
            
        # mask_anno = {}
        # for mask in tqdm(mask_list, desc='(Masks loading...) ', ascii=' ='):
        #     mask_object = nib.load(os.path.join(self.dir, person_id, 'nii', mask))
        #     mask_array = mask_object.get_fdata()
        #     mask_array = resize_img(mask_array, (64, 128, 128))
        #     mask_anno[mask.split('_')[0]] = mask_array

        cls, bbox, mask = [], [], []
        for tooth in tqdm(annos, desc='(Annotating...) ', ascii=' ='):
            cls.append(np.expand_dims((np.array(TOOTH_NUM)==tooth)*1, axis=0))
            
            box = bbox_anno[tooth]
            # bbox -> heatmap
            box = np.array([(box[0]+box[3])/2, (box[1]+box[4])/2, (box[2]+box[5])/2]) / np.array(img_array.shape) * np.array([64, 128, 128])
            heatmap = generate_gaussian_heatmap((64, 128, 128), box)
            bbox.append(np.expand_dims(heatmap, axis=0))
            
            mask_object = nib.load(os.path.join(self.dir, person_id, 'nii', tooth+"_gt.nii.gz"))
            mask_array = mask_object.get_fdata()
            mask_array = resize_img(mask_array, (64, 128, 128))
            mask.append(np.expand_dims(mask_array, axis=0))
            # else:
            #     bbox.append(None)
            #     mask.append(None)
        
        
        img_array = resize_img(np.array(img_array), (64, 128, 128))
        np_cls = np.concatenate(cls, axis=0)
        np_bbox = np.concatenate(bbox, axis=0)
        np_mask = np.concatenate(mask, axis=0)
        
        
        proccessed_out = {
            'person_id': person_id,
            'image': img_array,
            'cls': np_cls,
            'bbox': np_bbox,
            'mask' : np_mask
        }
        
        proccessed_out = self.transform(proccessed_out)
        
        return proccessed_out
        # return None



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