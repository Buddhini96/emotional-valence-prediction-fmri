import numpy as np
import os
from scipy.io import loadmat
import pandas as pd

from python_scripts.ml_models.model_trainer import get_test_set_accuracy
from python_scripts.utils.file_processor import get_nii_data, read_ds00025_event_file

NO_OF_TRIALS = 64
NO_OF_NUISSANCE_REGRESSORS = 6
NO_OF_RUNS = 2
CV_FOLDS = 3
VOXELS = [100, 400, 800, 1600, 2400, 3200]

result_sheet = {"Classifier": [],
                "Metric": []}
ML_MODELS = ["SVM", "GNB", "RF", "LR"]

#subjects = ["01", "02"]
subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]


for no_of_voxels in VOXELS:

    for turn, subject in enumerate(subjects):
        subject_name = f"sub-{subject}"
        if subject_name not in result_sheet:
            result_sheet[subject_name] = []

        base_path = f"E:\\ds000205\\sub-{subject}\\GLMNormal\\"
        design_mat_file_path = f"E:\\ds000205\\sub-{subject}\\GLMNormal\\SPM.mat"

        sub_trial_types = read_ds00025_event_file(split_runs=True)

        data = loadmat(design_mat_file_path)

        targets = [[0 for _ in range(NO_OF_TRIALS + NO_OF_NUISSANCE_REGRESSORS)] for i in range(NO_OF_RUNS)]

        for i, data_block in enumerate(data['SPM']['Vbeta'][0][0][0]):
            list_index = i // (NO_OF_TRIALS + NO_OF_NUISSANCE_REGRESSORS)
            if list_index < len(targets):
                targets[list_index][i % (NO_OF_TRIALS + NO_OF_NUISSANCE_REGRESSORS)] = data_block[0][0]

        X_file_names = []
        Y = []

        for i in range(NO_OF_RUNS):
            for j in range(NO_OF_TRIALS):
                X_file_names.append(str(targets[i][j]))
                Y.append(sub_trial_types[i][j])

        X = [get_nii_data(base_path + file_name) for file_name in X_file_names]

        for model in ML_MODELS:
            result = get_test_set_accuracy(X, Y, model, CV_FOLDS, no_of_voxels)
            for metric, value in result.items():
                if turn == 0:
                    result_sheet["Classifier"].append(f"{model}-({no_of_voxels})")
                    result_sheet["Metric"].append(metric)
                result_sheet[subject_name].append(value)

        print(f"{subject_name} results have been finalized")

print(result_sheet)

df = pd.DataFrame(result_sheet)

df.to_csv("../../output/ds00025_glm_normal_results.csv")