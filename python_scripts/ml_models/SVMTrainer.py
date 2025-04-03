from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

from python_scripts.utils.results_generator import calculate_accuracy


class SVMTrainer:
    
    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = SVC(probability=True,random_state=42)
        
    def train_model(self):
        param_distributions = {
        'C': loguniform(1e-4, 1e2),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': randint(1, 5),
        'gamma': ['scale', 'auto'] + list(loguniform(1e-4, 1e2).rvs(15)),
        'coef0': loguniform(1e-4, 1e1),
        'shrinking': [True, False],
        'class_weight': [None, 'balanced'],
        'probability': [True]
        }
        # param_distributions = {
        #     'C': loguniform(1e-3, 10),  # Reduced range
        #     'kernel': ['linear', 'rbf'],  # Limited to faster kernels
        #     'gamma': ['scale', 'auto', loguniform(1e-3, 10)],  # Smaller range
        #     'shrinking': [True],  # Avoiding 'False' to leverage the heuristic
        #     'class_weight': [None, 'balanced'],
        #     'probability': [False]  # Disabled probability estimation for speed
        # }
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=param_distributions, n_iter=20, cv=5, n_jobs=-1, random_state=42)
        random_search.fit(self.X_train, self.Y_train)
        y_pred = random_search.predict(self.X_test)
        return calculate_accuracy(self.Y_test, y_pred)
        # print(y_pred)
        # results = generate_results(self.Y_test, y_pred)
        # return results
        #return {"accuracy": random_search.best_score_}



