from lightgbm import LGBMClassifier
from results_generator import generate_results
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

class LGBMTrainer:
    
    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = LGBMClassifier(random_state=42)
    
    def train_model(self):
        param_distributions = {
            'n_estimators': randint(50, 500),
            'learning_rate': loguniform(1e-3, 1),
            'num_leaves': randint(10, 100),
            'subsample': loguniform(0.5, 1),
            'colsample_bytree': loguniform(0.5, 1)
        }
        random_search = RandomizedSearchCV(self.model, param_distributions, n_iter=50, cv=5, n_jobs=-1, random_state=42)
        random_search.fit(self.X_train, self.Y_train)
        y_pred = random_search.predict(self.X_test)
        return generate_results(self.Y_test, y_pred)