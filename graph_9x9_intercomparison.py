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
    dice_coronal = [dice_coefficient(gt[i, :, :], seg[i, :, :]) for i in range(gt.shape[0])]
    dice_axial = [dice_coefficient(gt[:, i, :], seg[:, i, :]) for i in range(gt.shape[1])]
    dice_sagittal = [dice_coefficient(gt[:, :, i], seg[:, :, i]) for i in range(gt.shape[2])]
    return np.mean(dice_axial), np.mean(dice_coronal), np.mean(dice_sagittal), dice_axial, dice_coronal, dice_sagittal



# Axial
plane_names = ["Coronal", "Axial", "Sagittal"]
average_dices1 = {plane: [] for plane in plane_names}
per_slice_dices1 = {plane: [] for plane in plane_names}

average_dices2 = {plane: [] for plane in plane_names}
per_slice_dices2 = {plane: [] for plane in plane_names}

# MedSAM segmented image:
path_medsam_seg1 = 'data/saved_segs/0.01_error_000.nrrd'
path_medsam_seg2 = 'data/saved_segs/Segmentations_sample0.nrrd'
segmentations1, header1 = nrrd.read(path_medsam_seg1)
segmentations2, header2 = nrrd.read(path_medsam_seg2)

# Ground truth segmentation: 
# Labels: 3 = left lung, 4 = right lung, 5 = heart, 8 = bone
label000 = 'data/Dataset002_Version1_FullMouse\labelsTr\Mouse_labelled_000.nii.gz'
labeled_img = np.transpose(load_data(label000), (2, 1, 0))

print(segmentations1.shape)
print(segmentations2.shape)
print(labeled_img.shape)

# Read labels:
organ_labels_str1 = header1['organ_info']
organ_labels_str2 = header2['organ_info']

# Convert the string to a dictionary
organ_labels1 = ast.literal_eval(organ_labels_str1)
organ_labels2 = ast.literal_eval(organ_labels_str2)


for idx, (key, value) in enumerate(organ_labels1.items()):
    organ_name = value['name']
    seg1 = segmentations1[:, :, :, idx]
    seg2 = segmentations2[:, :, :, idx]
    
    if organ_name == 'Heart':
        selected_label = 5
    elif organ_name == 'Left lung':
        selected_label = 3
    elif organ_name == 'Right lung':
        selected_label = 4
    print(selected_label)
    gt = (labeled_img == selected_label).astype(np.uint8)
    print('Organ: ', organ_name)
    
    avg_dice_axial1, avg_dice_coronal1, avg_dice_sagittal1, dice_axial1, dice_coronal1, dice_sagittal1 = compute_dice_by_plane(gt, seg1)
    avg_dice_axial2, avg_dice_coronal2, avg_dice_sagittal2, dice_axial2, dice_coronal2, dice_sagittal2 = compute_dice_by_plane(gt, seg2)

    
    average_dices1["Axial"].append(avg_dice_axial1)
    average_dices1["Coronal"].append(avg_dice_coronal1)
    average_dices1["Sagittal"].append(avg_dice_sagittal1)
    
    per_slice_dices1["Axial"].append(dice_axial1)
    per_slice_dices1["Coronal"].append(dice_coronal1)
    per_slice_dices1["Sagittal"].append(dice_sagittal1)
    
    average_dices2["Axial"].append(avg_dice_axial2)
    average_dices2["Coronal"].append(avg_dice_coronal2)
    average_dices2["Sagittal"].append(avg_dice_sagittal2)
    
    per_slice_dices2["Axial"].append(dice_axial2)
    per_slice_dices2["Coronal"].append(dice_coronal2)
    per_slice_dices2["Sagittal"].append(dice_sagittal2)
    
    
        
# Create subplots (1 row for each plane)
fig, axes = plt.subplots(3, 3, figsize=(18, 6), sharey=True)

# Iterate through planes and plot Dice coefficients per slice for all organs
for i, plane in enumerate(plane_names):
    
    # Iterate through all organs
    for idx, organ_names in enumerate(organ_labels1.values()):
        ax = axes[i, idx]
        organ_name = organ_names['name']
        print(plane)
        organ_dice1 = per_slice_dices1[plane][idx]  # Dice coefficients per slice for this organ        
        organ_dice2 = per_slice_dices2[plane][idx]
        
        # Plot the Dice coefficients 
        ax.plot(range(len(organ_dice1)), organ_dice1, label='subject1')
        ax.plot(range(len(organ_dice2)), organ_dice2, label='subject2')
        
        if organ_name == 'Heart':
            selected_label = 5
        elif organ_name == 'Left lung':
            selected_label = 3
        elif organ_name == 'Right lung':
            selected_label = 4
        gt = (labeled_img == selected_label).astype(np.uint8)
        
        # Compute the range of slices where GT mask is non-zero
        if plane == "Sagittal":
            gt_nonzero = np.any(gt, axis=(0, 1))  # Non-zero slices along Sagittal (z-axis)
        elif plane == "Axial":
            gt_nonzero = np.any(gt, axis=(0, 2))  # Non-zero slices along Axial (y-axis)
        elif plane == "Coronal":
            gt_nonzero = np.any(gt, axis=(1, 2))  # Non-zero slices along Coronal (x-axis)
        else:
            raise ValueError("Invalid plane specified: Choose 'Sagittal', 'Axial', or 'Coronal'.")

        # Find all non-zero slice indices
        non_zero_indices = np.where(gt_nonzero)[0]
        if len(non_zero_indices) == 0:
            raise ValueError("No non-zero slices found!")

        # Compute first and last non-zero slice indices
        min_idx = non_zero_indices[0]
        max_idx = non_zero_indices[-1]

        print(f"First non-zero slice: {min_idx}, Last non-zero slice: {max_idx}")
        
        # Add vertical lines for min and max indices
        ax.axvline(x=min_idx, color='red', linestyle='--')
        ax.axvline(x=max_idx, color='red', linestyle='--')
        
        margin = 20
        start = max(min_idx - margin, 0)
        end = min(max_idx + margin, len(organ_dice1) - 1)
        
        # Set individual subplot titles and labels
        if i == 0:  # Add column title (organ name) for the top row
            ax.set_title(organ_name, fontsize=12)
        if idx == 0:  # Add row title (plane name) for the leftmost column
            ax.set_ylabel(f'{plane} Plane \n Dice Coefficient', fontsize=12)
        ax.set_xlabel('Slice Index', fontsize=10)
        ax.grid()

    
# Adjust layout and display
fig.suptitle('Dice Coefficients Per Slice for Each Plane', fontsize=16)
plt.legend(['subject1', 'subject2'])
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
plt.show()
