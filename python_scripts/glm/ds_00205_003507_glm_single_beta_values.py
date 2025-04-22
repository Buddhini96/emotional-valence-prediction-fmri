
from collections import defaultdict
import json
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from python_scripts.ml_models.model_trainer import get_test_set_accuracy, get_test_set_accuracy_across
from python_scripts.utils.file_processor import read_ds003507_event_file, read_ds00025_event_file

subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18",
            "19", "20", "21"]
CV_FOLDS = 8
result_sheet = {"Classifier": [],
                "Metric": []}
#ML_MODELS = ["SVM", "GNB", "RF", "LR"]
ML_MODELS = ["GNB", "RF", "LR"]

X_all = []
Y_all = []

with open('ds03507_required_indices.json', 'r') as file:
    loaded_data = json.load(file)

required_indices = loaded_data["indices"]
no_of_voxels = len(required_indices)

for turn, subject in enumerate(subjects):

    subject_name = f"sub-{subject}"
    mat_file_path = f"F:\\ds003507\\ds003507\\{subject_name}\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
    Y = read_ds003507_event_file(subject_name, False)
    file = h5py.File(mat_file_path, 'r')
    X = [np.nan_to_num(x.flatten()) for x in file["modelmd"]]

    X_filtered = [x for i, x in enumerate(X) if Y[i] in {1, 4}]
    Y_filtered = [y % 4 for y in Y if y in {1, 4}]

    X_filtered = [x[required_indices] for x in X_filtered]

    X_all.extend(X_filtered)
    Y_all.extend(Y_filtered)

print("Training GNB Model")
gnb_model = GaussianNB()
gnb_model.fit(X_all, Y_all)
print("Training RF Model")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2)
rf_model.fit(X_all, Y_all)
print("Training LR Model")
lr_model = LogisticRegression(solver='liblinear', C=0.007, penalty='l2', max_iter=1000, tol=0.0005, class_weight='balanced')
lr_model.fit(X_all, Y_all)        


ds00205_subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]

for turn, subject in enumerate(ds00205_subjects):
    subject_name = f"sub-{subject}"
    result_sheet[subject_name] = []
    mat_file_path = f"E:\\ds000205\\{subject_name}\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
    Y = read_ds00025_event_file(split_runs=False)
    file = h5py.File(mat_file_path, 'r')
    X = [np.nan_to_num(x.flatten()) for x in file["modelmd"]]
    print(f"length of {subject_name} required indices {len(required_indices)}")

    X = [x[required_indices] for x in X]

    y_pred_gnb = gnb_model.predict(X)
    y_pred_rf = rf_model.predict(X)
    y_pred_lr = lr_model.predict(X)

    if turn == 0:
        result_sheet["Classifier"].append(f"GNB-({no_of_voxels})")
        result_sheet["Metric"].append("accuracy")

        result_sheet["Classifier"].append(f"RF-({no_of_voxels})")
        result_sheet["Metric"].append("accuracy")

        result_sheet["Classifier"].append(f"LR-({no_of_voxels})")
        result_sheet["Metric"].append("accuracy")

    result_sheet[subject_name].append(accuracy_score(Y, y_pred_gnb))
    result_sheet[subject_name].append(accuracy_score(Y, y_pred_rf))
    result_sheet[subject_name].append(accuracy_score(Y, y_pred_lr))

    print(f"{subject_name} results have been finalized")

print(result_sheet)

df = pd.DataFrame(result_sheet)

df.to_csv("../../output/ds00025_across_data_sets_results.csv")


