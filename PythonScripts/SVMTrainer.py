from sklearn.svm import SVC
from results_generator import generate_results
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

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
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=param_distributions, n_iter=50, cv=3, n_jobs=-1, random_state=42)
        random_search.fit(self.X_train, self.Y_train)
        y_pred = random_search.predict(self.X_test)
        results = generate_results(self.Y_test, y_pred)
        return results