import numpy as np
from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from python_scripts.ml_models.GNBTrainer import GNBTrainer
from python_scripts.ml_models.LRTrainer import LRTrainer
from python_scripts.ml_models.RFTrainer import RFTrainer
from python_scripts.ml_models.SVMTrainer import SVMTrainer
import logging

# Configure logging
logging.basicConfig(filename='output.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def t_test_score(X, y):
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("The t-test can only be used for binary classification problems")

    group_scores = [X[y == c] for c in classes]
    t_stats, p_values = ttest_ind(group_scores[0], group_scores[1], axis=0)
    scores = np.abs(t_stats)
    return scores, np.ones_like(scores)

def train_model(model_name, X_train, X_test, Y_train, Y_test):
    trainers = {
        'LR': LRTrainer,
        'RF': RFTrainer,
        'SVM': SVMTrainer,
        'GNB': GNBTrainer
    }

    if model_name not in trainers:
        raise ValueError(f"Unsupported model name : {model_name}. \n Available models: 'linear_regression', 'random_forest', 'svm', 'GNB'")

    return trainers[model_name](X_train, X_test, Y_train, Y_test).train_model()

def select_features(X_train, y_train, X_test, no_of_voxels):
    selector = SelectKBest(score_func=t_test_score, k=no_of_voxels)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    return X_train, X_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()  # Or MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def get_test_set_accuracy(X, Y, model_name, cv_folds, no_of_voxels):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []
    X = np.array(X)
    Y = np.array(Y)

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        print(f"{model_name} is training by selecting {no_of_voxels} from {len(X_train)} length training set")
        print("Samples with trial type 1", len([x for x in y_train if x == 1]))
        print("Samples with trial type 0", len([x for x in y_train if x == 0]))

        if no_of_voxels:
            X_train, X_test = select_features(X_train, y_train, X_test, no_of_voxels)

        X_train, X_test = scale_data(X_train, X_test)
        accuracies.append(train_model(model_name, X_train, X_test, y_train, y_test))

    mean_accuracy = sum(accuracies)/len(accuracies)

    return {"accuracy": mean_accuracy}


def get_correlated_voxel_indices(X, stimulus_occurance_matrix):
    required_indices = []
    for i in range(len(X[0])):
        corr_coefficient = []
        avg_corr_coefficient = 0
        for stimulus_id, onset_ids in stimulus_occurance_matrix.items():
            beta_values = [X[onset_ids[0]][i], X[onset_ids[1]][i], X[onset_ids[2]][i], X[onset_ids[3]][i]]
            avg_pearson_coefficient = calculate_average_correlation(beta_values)
            if not np.isnan(avg_pearson_coefficient):
                corr_coefficient.append(avg_pearson_coefficient)
        if len(corr_coefficient) > 0:
            avg_corr_coefficient = sum(corr_coefficient)/len(corr_coefficient)
        if avg_corr_coefficient > 0.2:
            required_indices.append(i)
        logging.info(f"voxel {i} calculated, {corr_coefficient}, {avg_corr_coefficient}")
    return required_indices

def calculate_average_correlation(beta_values):
    # pearson_corr1 = np.corrcoef([beta_values[0], beta_values[1]], [beta_values[2], beta_values[3]])[0, 1]
    # pearson_corr2 = np.corrcoef([beta_values[0], beta_values[2]], [beta_values[1], beta_values[3]])[0, 1]
    # pearson_corr3 = np.corrcoef([beta_values[0], beta_values[3]], [beta_values[1], beta_values[2]])[0, 1]
    r1, p1 = pearsonr([beta_values[0], beta_values[1]], [beta_values[2], beta_values[3]])
    r2, p2 = pearsonr([beta_values[0], beta_values[2]], [beta_values[1], beta_values[3]])
    r3, p3 = pearsonr([beta_values[0], beta_values[3]], [beta_values[1], beta_values[2]])

    pearsonrs = []

    if not np.isnan(r1): pearsonrs.append(r1)
    if not np.isnan(r2): pearsonrs.append(r2)
    if not np.isnan(r3): pearsonrs.append(r3)

    if len(pearsonrs) > 0: return sum(pearsonrs)/len(pearsonrs)

    return np.nan


