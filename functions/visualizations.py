import numpy as np
import cv2

slice_indx = 0

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
    data = param['data']
    global slice_indx
    
    if event == cv2.EVENT_MOUSEWHEEL: # Check if mousewheel has scrolled:
        if flags > 0: # If scroll is positive
            slice_indx += 1
        else: # If scroll is negative
            slice_indx -= 1
                # Ensure the slice index is within the valid range
        slice_indx = update_slice(slice_indx, data)