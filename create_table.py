import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import nrrd
sys.path.append('functions/')
from visualizations import *
import ast
import pandas as pd

def dice_coefficient(gt,seg):
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum((gt > 0) & (seg > 0))
    total = np.sum(gt > 0) + np.sum(seg > 0)
    return (2.0 * intersection / total) if total > 0 else 0.0

def compute_dice_by_plane(gr, seg):
    """Compute Dice coefficient averaged over slices for each plane."""
    dice_coronal = [dice_coefficient(gt[i, :, :], seg[i, :, :]) for i in range(gt.shape[0])]
    dice_axial = [dice_coefficient(gt[:, i, :], seg[:, i, :]) for i in range(gt.shape[1])]
    dice_sagittal = [dice_coefficient(gt[:, :, i], seg[:, :, i]) for i in range(gt.shape[2])]
    return np.mean(dice_axial), np.mean(dice_coronal), np.mean(dice_sagittal), dice_axial, dice_coronal, dice_sagittal

plane_names = ["Axial", "Coronal", "Sagittal"]
average_dices = {plane: [] for plane in plane_names}
per_slice_dices = {plane: [] for plane in plane_names}

# MedSAM segmented image:
headers = []
segmentations_all = []
segmentations_all_gt = []
for i in range(4):
    path_medsam_seg = f'data/saved_segs/0.01_error_00{i}.nrrd'
    print(path_medsam_seg)
    segmentations, header = nrrd.read(path_medsam_seg)
    segmentations_all.append(segmentations)
    headers.append(header)
    
    label000 = f'data/Dataset002_Version1_FullMouse\labelsTr\Mouse_labelled_00{i}.nii.gz'
    labeled_img = np.transpose(load_data(label000), (2, 1, 0))  
    
    segmentations_all_gt.append(labeled_img)
    
for i in range(4):
    path_medsam_seg = f'data/saved_segs/Segmentations_sample{i}.nrrd'
    segmentations, header = nrrd.read(path_medsam_seg)
    segmentations_all.append(segmentations)
    headers.append(header)

# Read labels:
organ_labels_str = headers[0]['organ_info']

# Convert the string to a dictionary
organ_labels = ast.literal_eval(organ_labels_str)

results = {value['name']: {'Average DSC': [], 'Max DSC': []} for _, value in organ_labels.items()}

for sample_idx, segmentations in enumerate(segmentations_all):
    labeled_array = segmentations_all_gt[sample_idx % 4]
    print(sample_idx)
    for idx, (key, value) in enumerate(organ_labels.items()):
        organ_name = value['name']
        seg = segmentations[:, :, :, idx]
        
        if organ_name == 'Heart':
            selected_label = 5
        if organ_name == 'Left lung':
            selected_label = 3
        if organ_name == 'Right lung':
            selected_label = 4
        
        gt = (labeled_array == selected_label).astype(np.uint8)
        print('Organ: ', organ_name)
        
        avg_dice_axial, avg_dice_coronal, avg_dice_sagittal, dice_axial, dice_coronal, dice_sagittal = compute_dice_by_plane(gt, seg)
        np.max(dice_axial)
        avg_dsc = dice_coefficient(gt, seg)
        max_dsc = np.max([np.max(dice_axial), np.max(dice_coronal), np.max(dice_sagittal)])
        
        results[organ_name]['Average DSC'].append(avg_dsc)
        results[organ_name]['Max DSC'].append(max_dsc)
        
        print('Average DSC: ', avg_dsc)
        print('Max DSC: ', max_dsc)

table_data = {
    'Organ': [],
    'Average DSC': [],
    'Max DSC': []
}

for organ_name, metrics in results.items():
    table_data["Organ"].append(organ_name)
    table_data["Average DSC"].append(np.mean(metrics["Average DSC"]))  # Average across samples
    table_data["Max DSC"].append(np.max(metrics["Max DSC"]))  # Max across samples

# Create a pandas DataFrame for visualization
df = pd.DataFrame(table_data)

# Display the table
print(df)