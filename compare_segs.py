import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import nrrd
sys.path.append('functions/')
from visualizations import *
import ast

# MedSAM segmented image:
path_medsam_seg = 'data/saved_segs/test1.nrrd'
segmentation, header = nrrd.read(path_medsam_seg)

# Ground truth segmentation: 
label000 = 'data/Dataset002_Version1_FullMouse\labelsTr\Mouse_labelled_000.nii.gz'
labeled_img = load_data(label000)

# Read labels:
organ_labels_str = header['organ_labels']

# Convert the string to a dictionary
organ_labels = ast.literal_eval(organ_labels_str)

print("Parsed dictionary:", organ_labels[1])
