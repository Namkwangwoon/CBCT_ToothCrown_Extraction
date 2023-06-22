import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 파일 불러오기
# nii_img = nib.load('/media/jeeheon/SSD/Dataset_osstem_toothseg/2차/1/1/nii/1.nii.gz')
nii_img = nib.load('/media/jeeheon/SSD/ToothFairy_Dataset/nii/val/P45/data.nii.gz')
# nii_img = nib.load('/home/jeeheon/Documents/3D-UNet/results/target_4.nii.gz')

# 데이터 배열 불러오기
img_data = nii_img.get_fdata()

# img_data = img_data.transpose(2, 1, 0)
# img_data = np.swapaxes(img_data,0,2)

# x축 방향으로 Maximum Intensity Projection 수행
mip_img_x = np.max(img_data, axis=0)
mip_img_y = np.max(img_data, axis=1)
mip_img_z = np.max(img_data, axis=2)

# 결과 확인
plt.imshow(mip_img_x, cmap='gray')
plt.show()

plt.imshow(mip_img_y, cmap='gray')
plt.show()

plt.imshow(mip_img_z, cmap='gray')
plt.show()