from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import ast
import sys
import nrrd
sys.path.append('functions/')
from visualizations import *

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

def compute_precision_reacll_by_plane(gr, seg):
    """Compute Dice coefficient averaged over slices for each plane."""
    print(gt[0, :, :])
    print(precision_recall(gt[0, :, :], seg[0, :, :]))
    
    metric_coronal = [precision_recall(gt[i, :, :], seg[i, :, :]) for i in range(gt.shape[0])]
    precision_coronal = [metric[0] for metric in metric_coronal]
    recall_coronal = [metric[1] for metric in metric_coronal]
    
    metric_axial = [precision_recall(gt[:, i, :], seg[:, i, :]) for i in range(gt.shape[1])]
    precision_axial = [metric[0] for metric in metric_axial]
    recall_axial = [metric[1] for metric in metric_axial]
    
    metric_sagittal = [precision_recall(gt[:, :, i], seg[:, :, i]) for i in range(gt.shape[2])]
    precision_sagittal = [metric[0] for metric in metric_sagittal]
    recall_sagittal = [metric[1] for metric in metric_sagittal]
    
    return np.mean(precision_coronal), np.mean(precision_axial), np.mean(precision_sagittal), np.mean(recall_coronal), np.mean(recall_axial), np.mean(recall_sagittal)


def precision_recall(gt, seg):
    """Compute precision and recall between two binary masks."""
    true_positive = np.sum((gt > 0) & (seg > 0))
    predicted_positive = np.sum(seg > 0)
    actual_positive = np.sum(gt > 0)
    
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / actual_positive if actual_positive > 0 else 0.0
    
    return precision, recall

plane_names = ["Axial", "Coronal", "Sagittal"]
average_dices = {plane: [] for plane in plane_names}
per_slice_dices = {plane: [] for plane in plane_names}

# MedSAM segmented image collect all images::
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
    print(path_medsam_seg)
    segmentations, header = nrrd.read(path_medsam_seg)
    segmentations_all.append(segmentations)
    headers.append(header)

# Read labels:
organ_labels_str = headers[0]['organ_info']

# Convert the string to a dictionary
organ_labels = ast.literal_eval(organ_labels_str)

results = {value['name']: {'Average DSC': [], 'Max DSC': []} for _, value in organ_labels.items()}

# Collect metrics
organ_metrics_accumulated = {value['name']: {'Dice': [], 'Precision': [], 'Recall': []} for _, value in organ_labels.items()}

for sample_idx, segmentations in enumerate(segmentations_all):
    labeled_array = segmentations_all_gt[sample_idx % 4]
    print(sample_idx)
    for idx, (key, value) in enumerate(organ_labels.items()):
        organ_name = value['name']
        print(organ_name)

        seg = segmentations[:, :, :, idx]
        
        if organ_name == 'Heart':
            selected_label = 5
        elif organ_name == 'Left lung':
            selected_label = 3
        elif organ_name == 'Right lung':
            selected_label = 4
   
        gt = (labeled_array == selected_label).astype(np.uint8)
        
        avg_dice_axial, avg_dice_coronal, avg_dice_sagittal, dice_axial, dice_coronal, dice_sagittal = compute_dice_by_plane(gt, seg)
        avg_precision_axial, avg_precision_coronal, avg_precision_sagittal, avg_recall_axial, avg_recall_coronal, avg_recall_sagittal = compute_precision_reacll_by_plane(gt, seg)
        
        #avg_dice = np.mean([avg_dice_axial, avg_dice_coronal, avg_dice_sagittal])
        #avg_precision = np.mean([avg_precision_axial, avg_precision_coronal, avg_precision_sagittal])
        #avg_recall = np.mean([avg_recall_axial, avg_recall_coronal, avg_recall_sagittal])
        #print(avg_dice, avg_precision, avg_recall)

        avg_dice = dice_coefficient(gt, seg)
        avg_precision, avg_recall = precision_recall(gt, seg)
        print(avg_dice, avg_precision, avg_recall)
        
        # Accumulate metrics
        organ_metrics_accumulated[organ_name]['Dice'].append(avg_dice)
        organ_metrics_accumulated[organ_name]['Precision'].append(avg_precision)
        organ_metrics_accumulated[organ_name]['Recall'].append(avg_recall)

# Compute average metrics for each organ
average_metrics = {}
for organ_name, metrics in organ_metrics_accumulated.items():
    average_metrics[organ_name] = {
        'Dice': np.mean(metrics['Dice']),
        'Precision': np.mean(metrics['Precision']),
        'Recall': np.mean(metrics['Recall'])
    }

# Visualization: Metrics per organ
fig, axes = plt.subplots(1, len(average_metrics), figsize=(9, 6), sharey=True)

for i, (organ_name, metrics) in enumerate(average_metrics.items()):
    print(organ_name)
    print(i)
    ax = axes[i]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    ax.bar(metric_names, metric_values, color=['skyblue', 'orange', 'green'])
    ax.set_title(f'{organ_name}')
    
    max_metric_value = max(metric_values)
    ax.set_ylim(0, max_metric_value * 1.2)  # Adjust based on ASSD range if necessary
    ax.set_ylabel('Metric Value')
    
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.grid(axis='y')

fig.suptitle('Segmentation Metrics for Each Organ', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()