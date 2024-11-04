import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import torch
from segment_anything import sam_model_registry
import sys
sys.path.append('functions/')
from utils.demo import BboxPromptDemo

MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"

# device = "cuda:0"

device = torch.device('cpu')        #Changed it to CPU to work on our laptops
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

#Load data
import numpy as np
import nrrd
import matplotlib.pyplot as plt

# path_data = 'C:/Users/20202932/8FM30_Klinischemodule1/KM-II-project-1/data/'
path_data = 'data/'
# Read the NRRD file
# readdata, header = nrrd.read(path_data + 'Baseline 15-05-2020 (0) - Mouse 5 - Separate Segmentations Muscle (1)/Baseline 15-05-2020 (0) - Mouse 5 - Separate Segmentations Muscle/50kVp_0000.dcm.nrrd')
readdata, header = nrrd.read(path_data + '50kVp_0000.dcm.nrrd')

slice_index = 100
image = readdata[:,  slice_index, :]

# Convert grayscale image to RGB image for the SAM model
if len(image.shape) == 2:
    image = np.stack([image]*3, axis=-1)

# Scale the pixel values to [0, 1]
# First calculate the min and max value of image
old_min, old_max = image.min(), image.max()

# Then scale
image = (image - old_min) / (old_max - old_min)

# Clip the values to ensure they are between 0 and 1
slice_rgb = np.clip(image, 0, 1)
print(slice_rgb)

bbox_prompt_demo = BboxPromptDemo(medsam_model)
bbox_prompt_demo.show(slice_rgb)

