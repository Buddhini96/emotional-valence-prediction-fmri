import nibabel as nib

# Load the NIfTI file
#nii_file = nib.load("F:\\ds003507\\sub-01\\firstLevelAnalysisfromAFNI\\beta_0001.nii")
# nii_file = nib.load("F:\\ds003507\\sub1\\download\\run1\\r1_scale.nii")

#contrast_map = nib.load("F:\\ds003507\\sub-07\\firstLevelAnalysis\\beta_0001.nii")

# Get the voxel sizes (pixel dimensions in mm)
# voxel_sizes = nii_file.header.get_zooms()
#
# print("Voxel sizes (in mm):", voxel_sizes)
#
# nii_file1 = nib.load("E:\\ds000205-fmriprep\\sub-01\\func\\run1\\data\\sub-01_task-view_run-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii")
# voxel_sizes1 = nii_file1.header.get_zooms()
#
# print("Voxel sizes (in mm):", voxel_sizes1)
#
# nii_file2 = nib.load("E:\\ds000205-fmriprep\\sub-02\\func\\run1\\data\\resampled_sub-02_task-view_run-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii")
# voxel_sizes2 = nii_file2.header.get_zooms()

nii_file3 = nib.load("E:\ds000205-fmriprep\sub-01\\rsub-01_task-view_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii")
voxel_sizes3 = nii_file3.header.get_zooms()
print("Voxel sizes (in mm):", voxel_sizes3)