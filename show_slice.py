import numpy as np
import nrrd
import slicerio
import cv2
import sys

sys.path.append('functions/')
from visualizations import *

# Select example image -> reads image as numpy array
readdata, header = nrrd.read('data/50kVp_0000.dcm.nrrd')
segmentation = slicerio.read_segmentation('data/Segmentation.seg.nrrd')
slice_indx = 0

# Clip the data and set range from 0-1
clipped_data = map_range(np.clip(readdata, 0, 850), 0, 850, 0, 1)

callback_data = {'data': clipped_data}

cv2.namedWindow('CT-scan Viewer')
cv2.setMouseCallback('CT-scan Viewer', on_scroll, callback_data)
# slice_indx = update_slice(slice_indx, clipped_data)

cv2.waitKey(0)
cv2.destroyAllWindows()


