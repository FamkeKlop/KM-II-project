import numpy as np
import nrrd
import slicerio
import cv2

def map_range(values, old_min, old_max, new_min, new_max):
    """
    Maps from one range to another
    
    Param values: data to be mapped
    Param old_min: min value of old range
    Param old_max: max value of old range
    Param new_min: min value of new range
    Param new_max: max value of new range
    """
    # Scale values from the old range to the new range
    return (values - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

def update_slice(slice_indx, data):
    """
    Updates the slice
    
    Param slice_indx: current slice
    """
    # Make sure that scrolling does not go out of bounds (if so it goes to closest boundary)
    slice_indx = np.clip(slice_indx, 0, data.shape[2] - 1) 
    
    # Show current image slice
    cv2.imshow('CT-scan Viewer',  data[:,:,slice_indx])
    return slice_indx

def on_scroll(event, x, y, flags, param):
    """
    Updates the slice
    """
    global slice_indx
    
    if event == cv2.EVENT_MOUSEWHEEL: # Check if mousewheel has scrolled:
        if flags > 0: # If scroll is positive
            slice_indx += 1
        else: # If scroll is negative
            slice_indx -= 1
                # Ensure the slice index is within the valid range
        slice_indx = update_slice(slice_indx, clipped_data)


# Select example image -> reads image as numpy array
readdata, header = nrrd.read('50kVp_0000.dcm.nrrd')
segmentation = slicerio.read_segmentation('Segmentation.seg.nrrd')
slice_indx = 0

# Clip the data and set range from 0-1
clipped_data = map_range(np.clip(readdata, 0, 850), 0, 850, 0, 1)

cv2.namedWindow('CT-scan Viewer')
cv2.setMouseCallback('CT-scan Viewer', on_scroll)
slice_indx = update_slice(slice_indx, clipped_data)

cv2.waitKey(0)
cv2.destroyAllWindows()


