import numpy as np

# Example: Your old_array and complete_array
old_array = np.array([[1, 0, 1],
                      [0, 1, 0],
                      [1, 1, 0]])

complete_array = np.array([[1, 1, 0],
                           [1, 0, 1],
                           [0, 1, 1]])

# Sum the arrays
composite_array = old_array + complete_array

# Convert 2s to 1s, 1s to 0s, and keep 0s as 0s
composite_array = (composite_array == 2).astype(int)

print(composite_array)