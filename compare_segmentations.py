import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import nrrd
sys.path.append('functions/')
from visualizations import *


label000 = 'data/Dataset002_Version1_FullMouse\labelsTr\Mouse_labelled_000.nii.gz'
image000 = 'data/Dataset002_Version1_FullMouse\imagesTr\Mouse_labelled_000_0000.nii.gz'
segmentation, header = nrrd.read('segs.nrrd')

labeled_img = load_data(label000)
img = load_data(image000)

# Labels: 3 = left lung, 4 = right lung, 5 = heart, 8 = bone

selected_label = 5
selected_gr = (labeled_img == selected_label).astype(np.uint8)
mask = np.ma.masked_where(selected_gr == 0, selected_gr)
segmentation_mask = np.ma.masked_where(segmentation == 0, segmentation)

dice_coeff = dice_coef(segmentation_mask, mask[:,:,100])
print('dice coefficient: ', dice_coeff)

plt.imshow(img[:,:,100], 'gray', interpolation = None)
plt.imshow(mask[:,:,100], 'winter', alpha=0.5, interpolation = None)
plt.imshow(segmentation_mask, interpolation = None, alpha=0.5)
plt.title('Heart')
plt.show()








"""
def read_gz(filename, dtype=np.float32, shape=None):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    if shape:
        data = data.reshape(shape)
    return data
"""

"""
#readdata1 = read_gz(label000)
#readdata2 = read_gz(image000)
print(readdata1.shape)
print(readdata2.shape)
print(readdata1 == readdata2)
"""