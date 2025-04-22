import h5py
import numpy as np
import os
import pandas as pd

from python_scripts.ml_models.model_trainer import get_test_set_accuracy
from python_scripts.utils.file_processor import read_ds003507_event_file, read_ds003507_required_event_file

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18",
            "19", "20", "21"]
NO_OF_COMPONENTS = [25, 50, 100]
ML_MODELS = ["SVM", "GNB", "RF", "LR"]
CV_FOLDS = 8
NO_OF_TRIALS = 56
NO_OF_NUISSANCE_REGRESSORS = 6

result_sheet = {"Classifier":[],
                "Metric":[]}

for no_of_component in NO_OF_COMPONENTS:

    for turn, subject in enumerate(SUBJECTS):

        subject_name = f"sub-{subject}"
        if subject_name not in result_sheet:
            result_sheet[subject_name] = []

        regressor_mat_file_path = f"F:\\ds003507\\ds003507\\sub-{subject}\\GICA_{no_of_component}\\infomax_icasso_temporal_regression.mat"

        file = h5py.File(regressor_mat_file_path, 'r')
        regressionParameters = file['regressInfo']['regressionParameters'][:]

        sub_xx_data = regressionParameters.T

        Y = read_ds003507_event_file(subject_name, split_runs=False)
        X = []

        for i, data_points in enumerate(sub_xx_data):
            index = i%(NO_OF_TRIALS + NO_OF_NUISSANCE_REGRESSORS + 1)
            if index<NO_OF_TRIALS:
                X.append(data_points)

        X_filtered = [x for i, x in enumerate(X) if Y[i] in {1, 4}]
        Y_filtered = [y % 4 for y in Y if y in {1, 4}]

        for model in ML_MODELS:
            result = get_test_set_accuracy(X_filtered, Y_filtered, model, CV_FOLDS, None)
            for metric, value in result.items():
                if turn == 0:
                    result_sheet["Classifier"].append(f"{model}-({no_of_component})")
                    result_sheet["Metric"].append(metric)
                result_sheet[subject_name].append(value)

        print(f"{subject_name} results have been finalized")


print(result_sheet)
df = pd.DataFrame(result_sheet)
df.to_csv("../../output/ds03507_ica_within_results.csv")