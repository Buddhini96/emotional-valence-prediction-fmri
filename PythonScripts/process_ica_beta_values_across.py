import h5py
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from GNBTrainer import GNBTrainer
from LRTrainer import LRTrainer
from RFTrainer import RFTrainer
from SVMTrainer import SVMTrainer
from XGBoostTrainer import XGBoostTrainer
from LGBMTrainer import LGBMTrainer
from CatBoostTrainer import CatBoostTrainer
from CNNTrainer import CNNTrainer

SUBJECT_NOS = ["01", "02", "03", "04", "05", "06", "16", "21"]
NO_OF_COMPONENTS = 25

result_sheet = {"Classifier":[],
                "Metric":[]}

total_X = {}
total_y = {}

for SUBJECT_NO in SUBJECT_NOS:
    tsv_folder_path = f"F:\\ds003507\\ds003507_required_filtered_files\\sub-{SUBJECT_NO}\\"
    regressor_mat_file_path = f"F:\\ds003507\\sub-{SUBJECT_NO}\\GICA_{NO_OF_COMPONENTS}\\infomax_regular_temporal_regression.mat"

    file = h5py.File(regressor_mat_file_path, 'r')
    regressionParameters = file['regressInfo']['regressionParameters'][:]

    sub_xx_data = regressionParameters.T

    sub_trial_types = []

    for filename in os.listdir(tsv_folder_path):
        file_path = os.path.join(tsv_folder_path, filename)
        data = np.loadtxt(file_path, delimiter='\t', dtype=str, skiprows=1)
        numeric_column = data[:, 2].astype(int)
        filtered_list = [int(x) for x in list(numeric_column)]
        print(len(filtered_list))
        sub_trial_types.extend(filtered_list)

    X = []
    y = [x if x==1 else 0 for x in sub_trial_types ]

    #np.savetxt(f'sub_{SUBJECT_NO}_data.txt', sub_xx_data, delimiter='\t', fmt='%.8f')

    for i, data_points in enumerate(sub_xx_data):
        index = i%31
        if index<24:
            X.append(data_points)
        print("Appending Data", SUBJECT_NO, len(data_points))

    total_X[SUBJECT_NO] = X
    total_y[SUBJECT_NO] = y

for turn, SUBJECT_NO in enumerate(SUBJECT_NOS):
    subject_name = f"sub-{SUBJECT_NO}"
    result_sheet[subject_name] = []

    X_train, X_test, y_train, y_test = [], [], [], []

    for subject, data in total_X.items():
        if subject != SUBJECT_NO:
            X_train.extend(data)
        else:
            X_test.extend(data)

    for subject, data in total_y.items():
        if subject != SUBJECT_NO:
            y_train.extend(data)
        else:
            y_test.extend(data)

    scaler = StandardScaler()  # Or MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Samples with trial type 1", len([x for x in y_train if x == 1]))
    print("Samples with trial type 4 (0)", len([x for x in y_train if x == 0]))

    results_dictionary = {}

    svm_model = SVMTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    svm_results = svm_model.train_model()
    results_dictionary["SVM Classifier"] = svm_results

    rf_model = RFTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    rf_results = rf_model.train_model()
    results_dictionary["RF Classifier"] = rf_results

    gnb_model = GNBTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    gnb_results = gnb_model.train_model()
    results_dictionary["GNB Classifier"] = gnb_results

    lr_model = LRTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    lr_results = lr_model.train_model()
    results_dictionary["LR Classifier"] = lr_results

    xgb_model = XGBoostTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    xgb_results = xgb_model.train_model()
    results_dictionary["XGBoost Classifier"] = xgb_results

    lgbm_model = LGBMTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    lgbm_results = lgbm_model.train_model()
    results_dictionary["LGBM Classifier"] = lgbm_results

    catboost_model = CatBoostTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    catboost_results = catboost_model.train_model()
    results_dictionary["Catboost Classifier"] = catboost_results

    cnn_model = CNNTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    cnn_results = cnn_model.train_model()
    results_dictionary["CNN Classifier"] = cnn_results

    for classifier, metrics in results_dictionary.items():
        for metric, value in metrics.items():
            if turn == 0:
                result_sheet["Classifier"].append(classifier)
                result_sheet["Metric"].append(metric)
            result_sheet[subject_name].append(value)

print(result_sheet)
df = pd.DataFrame(result_sheet)
df.to_csv(f"{NO_OF_COMPONENTS}_ica_results.csv")