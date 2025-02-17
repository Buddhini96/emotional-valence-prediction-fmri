import h5py

file_path = "F:\\GroupICATv4.0c_standalone_Win64\\GICA_on_AFNI\\run1\\infomax_regular_ica_parameter_info.mat"

file = h5py.File(file_path, 'r')

print(file)