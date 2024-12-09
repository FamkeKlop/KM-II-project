import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import nrrd
sys.path.append('functions/')
from visualizations import *
import ast

def dice_coefficient(gt,seg):
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum((gt > 0) & (seg > 0))
    total = np.sum(gt > 0) + np.sum(seg > 0)
    return (2.0 * intersection / total) if total > 0 else 0.0

def compute_dice_by_plane(gr, seg):
    """Compute Dice coefficient averaged over slices for each plane."""
    dice_sagittal = [dice_coefficient(gt[i, :, :], seg[i, :, :]) for i in range(gt.shape[0])]
    dice_axial = [dice_coefficient(gt[:, i, :], seg[:, i, :]) for i in range(gt.shape[1])]
    dice_coronal = [dice_coefficient(gt[:, :, i], seg[:, :, i]) for i in range(gt.shape[2])]
    return np.mean(dice_axial), np.mean(dice_coronal), np.mean(dice_sagittal), dice_axial, dice_coronal, dice_sagittal


labels_gt = {
        "background": 0,
        "Bone": 1,
	"Spinal Cord": 2,
        "Left Lung": 3,
        "Right Lung": 4,
        "Heart": 5,
        "SAT": 6,
        "VAT": 7,
        "Muscle": 8        
    },

plane_names = ["Axial", "Coronal", "Sagittal"]
average_dices = {plane: [] for plane in plane_names}
per_slice_dices = {plane: [] for plane in plane_names}

# MedSAM segmented image:
path_medsam_seg = 'data/saved_segs/0.01_error_000.nrrd'
segmentations, header = nrrd.read(path_medsam_seg)

# Ground truth segmentation: 
# Labels: 3 = left lung, 4 = right lung, 5 = heart, 8 = bone
label000 = 'data/Dataset002_Version1_FullMouse\labelsTr\Mouse_labelled_000.nii.gz'
labeled_img = np.transpose(load_data(label000), (2, 1, 0))

print(segmentations.shape)
print(labeled_img.shape)

# Read labels:
organ_labels_str = header['organ_info']

# Convert the string to a dictionary
organ_labels = ast.literal_eval(organ_labels_str)


for idx, (key, value) in enumerate(organ_labels.items()):
    organ_name = value['name']
    seg = segmentations[:, :, :, idx]
    
    if organ_name == 'Heart':
        selected_label = 5
    if organ_name == 'Left Lung':
        selected_label = 3
    if organ_name == 'Right Lung':
        selected_label = 4
    
    gt = (labeled_img == selected_label).astype(np.uint8)
    print('Organ: ', organ_name)
    
    avg_dice_axial, avg_dice_coronal, avg_dice_sagittal, dice_axial, dice_coronal, dice_sagittal = compute_dice_by_plane(gt, seg)
    
    average_dices["Axial"].append(avg_dice_axial)
    average_dices["Coronal"].append(avg_dice_coronal)
    average_dices["Sagittal"].append(avg_dice_sagittal)
    
    per_slice_dices["Axial"].append(dice_axial)
    per_slice_dices["Coronal"].append(dice_coronal)
    per_slice_dices["Sagittal"].append(dice_sagittal)
    
    for i, plane in enumerate(plane_names):
        ax = axes[i]
    
        
# Create subplots (1 row for each plane)
fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)

# Iterate through planes and plot Dice coefficients per slice for all organs
for i, plane in enumerate(plane_names):
    ax = axes[i]
    
    # Iterate through all organs
    for idx, organ_name in enumerate(organ_labels.values()):
        organ_dice = per_slice_dices[plane][idx]  # Dice coefficients per slice for this organ
        
        ax.plot(range(len(organ_dice)), organ_dice, label=organ_name['name'])
        
        
    ax.set_title(f'{plane} Plane')
    ax.set_xlabel('Slice Index')
    if i == 0:
        ax.set_ylabel('Dice Coefficient')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid()

# Adjust layout and display
fig.suptitle('Dice Coefficients Per Slice for Each Plane', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
plt.show()

# Plot average Dice coefficients per plane for each organ
# Create subplots (1 row for each plane)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for i, plane in enumerate(plane_names):
    ax = axes[i]
    plt.figure()
    for idx, organ_name in enumerate(organ_labels.values()):
        plt.plot([plane], average_dices[plane][idx], marker='o', label=organ_name['name'])
    plt.xlabel('Planes')
    plt.ylabel('Average Dice Coefficient')
    plt.title('Average Dice Coefficients Across Planes')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.show()
