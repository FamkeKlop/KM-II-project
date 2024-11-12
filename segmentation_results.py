import numpy as np
import nrrd
from matplotlib import pyplot as plt

ground_truth, header = nrrd.read('data/Segmentation.seg.nrrd')
original_image, header1 = nrrd.read('data/50kVp_0000.dcm.nrrd')
slice_index = 100
image = original_image[:,  slice_index, :]

#ground_truth_100 = ground_truth[:,100,:]

unque_labels = np.unique(ground_truth)
print(unque_labels)
# Lable corresponding to heart is 11
selected_label = 11
selected_gt = (ground_truth == selected_label).astype(np.uint8)
selected_gt_slice100 = selected_gt[:,slice_index,:]
 
segmentation, header = nrrd.read('segs.nrrd')


print('ground truth segmentation shape: ',selected_gt_slice100.shape)
print('created segmentation shape: ', segmentation.shape)
print('original image shape: ', image.shape)

mask_gt = selected_label_mask_slice_100 > 0
mask_pred = segmentation > 0

"""
plt.imshow(mask_gt, cmap='autumn')
plt.axis("off")
plt.imshow()
plt.show()

plt.imshow(mask_pred, cmap='gray')
plt.axis("off")
plt.show()
"""

# Prepare the original image for overlay
if original_image.ndim == 3 and original_image.shape[0] == 1:  # If it's a single channel
    image = np.squeeze(image)  # Remove singleton dimensions

# Normalize the original image for display
image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

# Convert the 2D image to a 3D RGB format
image_rgb = np.stack([image_normalized] * 3, axis=-1)

# Create color overlays for masks
overlay_gt = np.zeros_like(image_rgb)
overlay_gt[mask_gt] = [1, 0, 0]  # Red for ground truth

overlay_pred = np.zeros_like(image_rgb)
overlay_pred[mask_pred] = [0, 1, 0]  # Green for predicted segmentation

# Combine original image with overlays
combined_image = np.clip(image_rgb + overlay_gt * 0.5 + overlay_pred * 0.5, 0, 1)  # Adjust transparency

# Plotting
plt.figure(figsize=(10, 10))
plt.imshow(combined_image)
plt.axis("off")
plt.title("Overlay of Ground Truth and Predicted Segmentation")
plt.show()