import numpy as np

# Ground truth and predicted masks
gt = np.array([[1, 1, 0, 0], [0, 0, 0, 0]])
seg = np.array([[1, 1, 1, 0], [0, 0, 1, 1]])

# Calculate true positives and predicted positives
true_positive = np.sum((gt > 0) & (seg > 0))  # True Positives = 2
print("True Positives:", true_positive)
predicted_positive = np.sum(seg > 0)  # Predicted Positives = 5
print("Predicted Positives:", predicted_positive)
# Calculate precision
precision = true_positive / predicted_positive if predicted_positive > 0 else 0
print("Precision:", precision)  # Output: 0.5
