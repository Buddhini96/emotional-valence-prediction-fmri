import os
import h5py
import numpy as np
import pandas as pd

from python_scripts.ml_models.model_trainer import get_test_set_accuracy, get_correlated_voxel_indices
from python_scripts.utils.file_processor import read_ds00025_event_file, get_ds_00025_stimulus_occurance_matrix, \
    read_ds003507_event_file

VOXELS = [100, 400, 800, 1600, 2400, 3200]
#subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
subjects = ["01", "02", "03", "04", "05", "06", "07", "08"]
CV_FOLDS = 8
result_sheet = {"Classifier": [],
                "Metric": []}

ML_MODELS = ["SVM", "GNB", "RF", "LR"]

for no_of_voxels in VOXELS:
    for turn, subject in enumerate(subjects):
        subject_name = f"sub-{subject}"
        if subject_name not in result_sheet:
            result_sheet[subject_name] = []
        mat_file_path = f"F:\\ds003507\\ds003507\\{subject_name}\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"

        Y = read_ds003507_event_file(subject_name, False)

        file = h5py.File(mat_file_path, 'r')
        X = [np.nan_to_num(x.flatten()) for x in file["modelmd"]]

        X_filtered = [x for i, x in enumerate(X) if Y[i] in {1,4}]
        Y_filtered = [y%4 for y in Y if y in {1,4}]

        for model in ML_MODELS:
            result = get_test_set_accuracy(X_filtered, Y_filtered, model, CV_FOLDS, no_of_voxels)
            for metric, value in result.items():
                if turn == 0:
                    result_sheet["Classifier"].append(f"{model}-({no_of_voxels})")
                    result_sheet["Metric"].append(metric)
                result_sheet[subject_name].append(value)

        print(f"{subject_name} results have been finalized")

print(result_sheet)

df = pd.DataFrame(result_sheet)

df.to_csv("../../output/003507_glm_single_voxel_results.csv")