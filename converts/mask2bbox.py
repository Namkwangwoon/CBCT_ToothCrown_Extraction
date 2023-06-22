import os
import numpy as np
import nibabel as nib

from skimage.measure import label, regionprops

img_path = os.path.abspath('/home/jeeheon/Documents/3D-UNet/gt_tooth.nii.gz')
save_path = os.path.abspath('/home/jeeheon/Documents/3D-UNet/gt_tooth_with_box.nii.gz')

img_object = nib.load(img_path)
img_array = img_object.get_fdata()
print(img_array.shape)

img_array_for_box = np.asarray(img_array, dtype=int)
print(img_array_for_box.max())
print(img_array_for_box.min())

lbl = img_array_for_box
props = regionprops(lbl)
print(props)

for prop in props:
    # min_row, min_col, min_depth / max_row, max_col, max_depth5d
    print('Found bbox', prop.bbox)
    minx, miny, minz, maxx, maxy, maxz = prop.bbox
    
np_res = img_array

nii_res = nib.Nifti1Image(np_res, affine=np.eye(4))
nib.save(nii_res, save_path)