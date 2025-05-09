from catboost import CatBoostClassifier
from python_scripts.utils.results_generator import calculate_accuracy
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

class CatBoostTrainer:
    
    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = CatBoostClassifier(verbose=0, random_state=42)
    
    def train_model(self):
        param_distributions = {
            'iterations': randint(50, 500),
            'learning_rate': loguniform(1e-3, 1),
            'depth': randint(3, 10)
        }
        random_search = RandomizedSearchCV(self.model, param_distributions, n_iter=50, cv=5, n_jobs=-1, random_state=42)
        random_search.fit(self.X_train, self.Y_train)
        y_pred = random_search.predict(self.X_test)
        return calculate_accuracy(self.Y_test, y_pred)