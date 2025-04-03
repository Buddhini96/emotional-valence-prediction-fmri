import h5py
import numpy as np
# mat_file_path = "F:\\fMRIDataProcessing\\GLMSingle-Matlab\\GLMsingle\\matlab\\examples\\exampleCongruenceSPMoutput\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
# file = h5py.File(mat_file_path, 'r')
#
mat_file_path1 = f"E:\\ds000205\\sub-01\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
#mat_file_path1 = f"E:\\ds000205-fmriprep\\sub-01\\GLMsingle\\TYPEC_FITHRF_GLMDENOISE.mat"
file1 = h5py.File(mat_file_path1, 'r')

# mat_file_path2 = "F:\\fMRIDataProcessing\\GLMSingle-Matlab\\GLMsingle\\matlab\\examples\\exampleCongruenceSPMoutput2\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
# file2 = h5py.File(mat_file_path2, 'r')

# mat_file_path3 = "F:\\fMRIDataProcessing\\GLMSingle-Matlab\\GLMsingle\\matlab\\examples\\exampleCongruenceSPMoutput3\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
# file3 = h5py.File(mat_file_path3, 'r')

# X = [x[0].flatten() for x in file["modelmd"]]
#
X1 = [np.nan_to_num(x.flatten()) for x in file1["modelmd"]]
#
# X2 = [x[0].flatten() for x in file2["modelmd"]]

# X3 = [x[0].flatten() for x in file3["modelmd"]]

length = len(X1)
#
for i in range(length):
    #print( i, min(X[i]), min(X1[i]), min(X2[i]), min(X3[i]), max(X[i]), max(X1[i]), max(X2[i]), max(X3[i]))
    print(i, min(X1[i]), max(X1[i]))

# from scipy.io import loadmat
#
# spm1 = loadmat("F:\\fMRIDataProcessing\\GLMSingle-Matlab\\GLMsingle\\matlab\\examples\\congruence_data\\SPM.mat")
#
# spm2 = loadmat("F:\\fMRIDataProcessing\\GLMSingle-Matlab\\GLMsingle\\matlab\\examples\\congruence_data2\\SPM.mat")
#
# print(spm1, spm2)


