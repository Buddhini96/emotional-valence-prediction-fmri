import h5py
import numpy as np
import os
import pandas as pd

from python_scripts.ml_models.model_trainer import get_test_set_accuracy, scale_data, train_model
from python_scripts.utils.file_processor import read_ds00025_event_file

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
NO_OF_COMPONENTS = [25]
ML_MODELS = ["SVM", "GNB", "RF", "LR"]
CV_FOLDS = 5
NO_OF_TRIALS = 64
NO_OF_NUISSANCE_REGRESSORS = 6

result_sheet = {"Classifier":[],
                "Metric":[]}

Y = [read_ds00025_event_file(split_runs=False) for _ in range(len(SUBJECTS)) ]

for no_of_component in NO_OF_COMPONENTS:

    regressor_mat_file_path = f"E:\\ds000205\\group_analysis\\infomax_regular_temporal_regression.mat"
    file = h5py.File(regressor_mat_file_path, 'r')
    regressionParameters = file['regressInfo']['regressionParameters'][:]
    sub_xx_data = regressionParameters.T
    X = [[] for sub in range(len(SUBJECTS))]


    for i, data_points in enumerate(sub_xx_data):
        index = i % (NO_OF_TRIALS + 1)
        sub = i//(2*(NO_OF_TRIALS+1))
        if index < NO_OF_TRIALS:
            X[sub].append(data_points)

    for turn, subject in enumerate(SUBJECTS):

        subject_name = f"sub-{subject}"
        if subject_name not in result_sheet:
            result_sheet[subject_name] = []

        for model in ML_MODELS:
            X_test = X[turn]
            Y_test = Y[turn]
            X_train = []
            Y_train = []

            for i in range(len(X)):
                if i != turn:
                    X_train.extend(X[i])
                    Y_train.extend(Y[i])

            X_train, X_test = scale_data(X_train, X_test)
            accuracy = train_model(model, X_train, X_test, Y_train, Y_test)
            result = {"accuracy": accuracy}
            for metric, value in result.items():
                if turn == 0:
                    result_sheet["Classifier"].append(f"{model}-({no_of_component})")
                    result_sheet["Metric"].append(metric)
                result_sheet[subject_name].append(value)

        print(f"{subject_name} results have been finalized")


print(result_sheet)
df = pd.DataFrame(result_sheet)
df.to_csv("../../output/ds00025_ica_across_results.csv")