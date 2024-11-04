import numpy as np
import nrrd
import slicerio
import cv2
import sys

sys.path.append('functions/')
from visualizations import map_range, on_scroll # type: ignore


# Select example image -> reads image as numpy array
# readdata, header = nrrd.read('data/Baseline 15-05-2020 (0) - Mouse 5 - Separate Segmentations Muscle (1)/Baseline 15-05-2020 (0) - Mouse 5 - Separate Segmentations Muscle/50kVp_0000.dcm.nrrd')
# segmentation = slicerio.read_segmentation('data/Baseline 15-05-2020 (0) - Mouse 5 - Separate Segmentations Muscle (1)/Baseline 15-05-2020 (0) - Mouse 5 - Separate Segmentations Muscle/Segmentation.seg.nrrd')

readdata, header = nrrd.read('data/50kVp_0000.dcm.nrrd')
print(readdata)
slice_indx = 0

# Clip the data and set range from 0-1
clipped_data = map_range(np.clip(readdata, 0, 850), 0, 850, 0, 1)

callback_data = {'data': clipped_data}

cv2.namedWindow('CT-scan Viewer')
cv2.setMouseCallback('CT-scan Viewer', on_scroll, callback_data)
# slice_indx = update_slice(slice_indx, clipped_data)

cv2.waitKey(0)
cv2.destroyAllWindows()


