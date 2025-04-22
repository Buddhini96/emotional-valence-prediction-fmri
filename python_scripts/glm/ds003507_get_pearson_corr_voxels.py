import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import csv

from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectKBest

from python_scripts.ml_models.model_trainer import get_correlated_voxel_indices
from python_scripts.utils.file_processor import read_ds003507_event_file

subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18",
            "19", "20", "21"]

def t_test_score(X, y):
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("The t-test can only be used for binary classification problems")

    group_scores = [X[y == c] for c in classes]
    t_stats, p_values = ttest_ind(group_scores[0], group_scores[1], axis=0)
    scores = np.abs(t_stats)
    return scores, np.ones_like(scores)

def get_ds003507_stimulus_occurance_matrix(Y_filtered):
    stimulus_onset = defaultdict(list)
    for i, stimulus in enumerate(Y_filtered):
        stimulus_onset[stimulus].append(i)
    return stimulus_onset


with open('ds003507_topk_output.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    for turn, subject in enumerate(subjects):
        subject_name = f"sub-{subject}"
        mat_file_path = f"F:\\ds003507\\ds003507\\{subject_name}\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
        Y = read_ds003507_event_file(subject_name, False)
        file = h5py.File(mat_file_path, 'r')
        X = [np.nan_to_num(x.flatten()) for x in file["modelmd"]]

        X_filtered = [x for i, x in enumerate(X) if Y[i] in {1, 4}]
        Y_filtered = [y % 4 for y in Y if y in {1, 4}]

        selector = SelectKBest(score_func=t_test_score, k=5000)
        selector.fit(X_filtered, Y_filtered)
        required_indices = selector.get_support(indices=True)
        # stimulus_dict = get_ds003507_stimulus_occurance_matrix(Y_filtered)
        # required_indices = get_correlated_voxel_indices(X_filtered, stimulus_dict)

        print(len(required_indices))
        writer.writerow(required_indices)
