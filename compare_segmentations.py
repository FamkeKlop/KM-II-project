import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import nrrd
sys.path.append('functions/')
from visualizations import *

# Labeled segmentation image
label000 = 'data/Dataset002_Version1_FullMouse\labelsTr\Mouse_labelled_000.nii.gz'
# Image
image000 = 'data/Dataset002_Version1_FullMouse\imagesTr\Mouse_labelled_000_0000.nii.gz'
# Ground truth segmentation
segmentation, header = nrrd.read('segs.nrrd')

# Load the .nii.gz files
labeled_img = load_data(label000)
img = load_data(image000)

# Labels: 3 = left lung, 4 = right lung, 5 = heart, 8 = bone
selected_label = 5
selected_gr = (labeled_img == selected_label).astype(np.uint8)  #
mask = np.ma.masked_where(selected_gr == 0, selected_gr)
segmentation_mask = np.ma.masked_where(segmentation == 0, segmentation)

dice_coeff = dice_coef(segmentation_mask, mask[:,:,100])
print('dice coefficient: ', dice_coeff)

plt.imshow(img[:,:,100], 'gray', interpolation = None)
plt.imshow(mask[:,:,100], 'winter', alpha=0.5, interpolation = None)
plt.imshow(segmentation_mask, interpolation = None, alpha=0.5)
plt.title('Heart')
plt.show()
