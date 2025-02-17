import numpy as np
import os
from scipy.io import loadmat
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from GNBTrainer import GNBTrainer
from LRTrainer import LRTrainer
from RFTrainer import RFTrainer
from SVMTrainer import SVMTrainer


# from XGBoostTrainer import XGBoostTrainer
# from LGBMTrainer import LGBMTrainer
# from CatBoostTrainer import CatBoostTrainer
# from CNNTrainer import CNNTrainer

def get_nii_data(file_path, top_voxel_indices):
    nii_image = nib.load(file_path)
    beta_data = nii_image.get_fdata()
    beta_data = np.nan_to_num(beta_data)
    flattened_beta_data = beta_data.flatten()
    output = flattened_beta_data[top_voxel_indices]
    print(file_path, len(output))
    return output


No_OF_TOP_VOXELS = 100  # Number of top voxels to extract

# subjects = ["01", "02", "03", "04", "05", "06", "16", "21"]

subjects = ["01"]

result_sheet = {"Classifier": [],
                "Metric": []}

for turn, subject in enumerate(subjects):

    subject_name = f"sub-{subject}"
    result_sheet[subject_name] = []

    base_path = f"/home/ds/Buddhini/data/sub-{subject}/firstLevelAnalysis/"
    tsv_folder_path = f"/home/ds/Buddhini/data/sub-{subject}/tsv_files/"
    design_mat_file_path = f"/home/ds/Buddhini/data/sub-{subject}/firstLevelAnalysis/SPM.mat"
    t_map_file = f"/home/ds/Buddhini/data/sub-{subject}/spmT_0001.nii"

    NO_OF_TRIALS = 24
    NO_OF_NUISSANCE_REGRESSORS = 6
    NO_OF_RUNS = 3

    sub_trial_types = []
    top_voxel_indices = []

    contrast_map = nib.load(t_map_file)
    contrast_data = contrast_map.get_fdata()  # Extract voxel values
    flatten_contrast_data = contrast_data.flatten()
    abs_t_values = np.abs(flatten_contrast_data)
    sorted_indices = np.argsort(abs_t_values)[::-1]
    top_voxel_indices = sorted_indices[:No_OF_TOP_VOXELS]

    for filename in os.listdir(tsv_folder_path):
        file_path = os.path.join(tsv_folder_path, filename)
        data = np.loadtxt(file_path, delimiter='\t', dtype=str, skiprows=1)
        numeric_column = data[:, 2].astype(int)
        sub_trial_types.append([int(x) for x in list(numeric_column)])

    data = loadmat(design_mat_file_path)

    targets = [[0 for _ in range(NO_OF_TRIALS + NO_OF_NUISSANCE_REGRESSORS)] for i in range(NO_OF_RUNS)]

    for i, data_block in enumerate(data['SPM']['Vbeta'][0][0][0]):
        list_index = i // (NO_OF_TRIALS + NO_OF_NUISSANCE_REGRESSORS)
        if list_index < len(targets):
            targets[list_index][i % (NO_OF_TRIALS + NO_OF_NUISSANCE_REGRESSORS)] = data_block[0][0]
            # print(list_index, data_block[5][0])
        # else:
        #     print(data_block[5][0])

    X_file_names = []
    Y = []

    for i in range(NO_OF_RUNS):
        for j in range(NO_OF_TRIALS):
            if sub_trial_types[i][j] in {1, 4}:
                X_file_names.append(str(targets[i][j]))
                Y.append(sub_trial_types[i][j] if sub_trial_types[i][j] == 1 else 0)
                # print(targets[i][j], sub_trial_types[i][j])
            else:
                print("This could be an error")

    X = [get_nii_data(base_path + file_name, top_voxel_indices) for file_name in X_file_names]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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

    # xgb_model = XGBoostTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    # xgb_results = xgb_model.train_model()
    # results_dictionary["XGBoost Classifier"] = xgb_results

    # lgbm_model = LGBMTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    # lgbm_results = lgbm_model.train_model()
    # results_dictionary["LGBM Classifier"] = lgbm_results

    # catboost_model = CatBoostTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    # catboost_results = catboost_model.train_model()
    # results_dictionary["Catboost Classifier"] = catboost_results

    # cnn_model = CNNTrainer(X_train_scaled, X_test_scaled, y_train, y_test)
    # cnn_results = cnn_model.train_model()
    # results_dictionary["CNN Classifier"] = cnn_results

    print(f"{subject_name} results have been finalized")

    for classifier, metrics in results_dictionary.items():
        for metric, value in metrics.items():
            if turn == 0:
                result_sheet["Classifier"].append(classifier)
                result_sheet["Metric"].append(metric)
            result_sheet[subject_name].append(value)

print(result_sheet)

df = pd.DataFrame(result_sheet)

df.to_csv(f"{No_OF_TOP_VOXELS}_voxel_results.csv")