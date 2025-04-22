from collections import defaultdict

import h5py
import numpy as np
from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectKBest
from python_scripts.utils.file_processor import read_ds00025_event_file

subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]

indices_dict = defaultdict(int)

def t_test_score(X, y):
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("The t-test can only be used for binary classification problems")

    group_scores = [X[y == c] for c in classes]
    t_stats, p_values = ttest_ind(group_scores[0], group_scores[1], axis=0)
    scores = np.abs(t_stats)
    return scores, np.ones_like(scores)

for turn, subject in enumerate(subjects):
    mat_file_path = f"E:\\ds000205\\sub-{subject}\\GLMsingle\\TYPED_FITHRF_GLMDENOISE_RR.mat"
    Y = read_ds00025_event_file(split_runs=False)
    file = h5py.File(mat_file_path, 'r')
    X = [np.nan_to_num(x.flatten()) for x in file["modelmd"]]
    selector = SelectKBest(score_func=t_test_score, k=5000)
    selector.fit(X, Y)
    selected_indices = selector.get_support(indices=True)
    print(selected_indices)
    for indices in selected_indices:
        indices_dict[indices] += 1

subject_count_indices = defaultdict(list)

for idx, count in indices_dict.items():
    subject_count_indices[count].append(int(idx))

print(len(subject_count_indices))

print(subject_count_indices[3])

