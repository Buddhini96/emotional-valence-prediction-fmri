# import nibabel as nib
# import numpy as np
#
# # Load the .nii file
# file_path = "F:\\ds003507\\sub-01\\func\\sub-01_task-affect_run-1_bold.nii"  # Replace with your file path
# img = nib.load(file_path)
#
# # Get the TR from the header
# tr = img.header["pixdim"][4]  # TR is stored in the 4th dimension (time)
# print(f"Repetition Time (TR): {tr} seconds")
#
# smoothed_image = nib.load("F:\\ds003507\\sub-01\\func\\swrsub-01_task-affect_run-1_bold.nii")
#
#
# single_beta_weight = nib.load("F:\\ds003507\\sub-01\\firstLevelAnalysis\\beta_0001.nii")
#
# beta_data = single_beta_weight.get_fdata()
# beta_data = np.nan_to_num(beta_data)
# unique_values = np.unique(beta_data)
#
# print(len(unique_values))
# print("Unique values in beta map:", np.unique(beta_data))

# res_image = nib.load("F:\\ds003507\\sub-01\\firstLevelAnalysis\\Res_0001.nii")

# print("Analysis is completed")

import nibabel as nib
import numpy as np
import pandas as pd

# Load the SPM t-map (replace with your actual file)
t_map_file = "F:\\ds003507\\sub-01\\firstLevelAnalysisPosNegContrast\\spmT_0001.nii"
img = nib.load(t_map_file)
data = img.get_fdata()  # Extract voxel values
affine = img.affine  # Transformation matrix


N = 100  # Number of top voxels to extract

# Flatten the 3D array and sort by T-values
flat_data = data.flatten()
sorted_indices = np.argsort(flat_data)[::-1]  # Descending order

# Extract top N indices and corresponding T-values
top_indices = sorted_indices[:N]
top_values = flat_data[top_indices]

print("Analysis Complete")