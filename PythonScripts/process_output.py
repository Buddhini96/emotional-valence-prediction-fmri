import nibabel as nib
import matplotlib.pyplot as plt

import SimpleITK as sitk
import matplotlib.pyplot as plt

# # Load and visualize
# img = sitk.ReadImage('F:\\GroupICATv4.0c_standalone_Win64\\new_test\\output\\test_sub01_timecourses_ica_s1_.nii')
# img_array = sitk.GetArrayFromImage(img)
# print(img_array.shape)
# plt.imshow(img_array, cmap='gray')
# plt.axis('off')
# plt.show()



# Load the NIfTI file
img = nib.load('F:\\GroupICATv4.0c_standalone_Win64\\new_test\\input\\sub_01_run_01\\sub-01_task-affect_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii')
data = img.get_fdata()

# Get image metadata
print("Initial Image shape:", data.shape)
print("Voxel dimensions:", img.header.get_zooms())

# Load the NIfTI file
img = nib.load('F:\\GroupICATv4.0c_standalone_Win64\\single_subject_test_new\\output_hrf_6_new\\infomax_regular_sub01_component_ica_s1_.nii')
data = img.get_fdata()

# Get image metadata
print("Subject Component shape:", data.shape)
print("Voxel dimensions:", img.header.get_zooms())

img = nib.load('F:\\GroupICATv4.0c_standalone_Win64\\single_subject_test_new\\output_hrf_6_new\\infomax_regular_sub01_timecourses_ica_s1_.nii')
data = img.get_fdata()

# Get image metadata
print("Time courses shape:", data.shape)
print("Voxel dimensions:", img.header.get_zooms())

# # Display a single slice along the z-axis (e.g., the middle slice)
# plt.imshow(data[:, :, data.shape[2] // 2], cmap='gray')
# plt.axis('off')
# plt.show()

img = nib.load('F:\\GroupICATv4.0c_standalone_Win64\\single_subject_test_new\\output_hrf_6_new\\infomax_regular_ica_subject_loadings.nii')
data = img.get_fdata()

# Get image metadata
print("Subject loadings shape:", data.shape)
print("Voxel dimensions:", img.header.get_zooms())



