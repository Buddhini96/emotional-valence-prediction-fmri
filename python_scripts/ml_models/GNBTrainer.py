from sklearn.naive_bayes import GaussianNB
from python_scripts.utils.results_generator import calculate_accuracy


class GNBTrainer:

    
    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = GaussianNB()
        
    def train_model(self):
        # param_distributions = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
        # random_search = RandomizedSearchCV(estimator=self.model, n_iter=20,
        #                                    param_distributions=param_distributions,
        #                                    cv=3, n_jobs=-1, random_state=42)
        # random_search.fit(self.X_train, self.Y_train)
        # y_pred = random_search.predict(self.X_test)
        # print(y_pred)
        # results = generate_results(self.Y_test, y_pred)
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # scores = cross_val_score(self.model, self.X_train, self.Y_train, cv=cv, scoring='accuracy')

        self.model.fit(self.X_train, self.Y_train)
        y_pred = self.model.predict(self.X_test)
        return calculate_accuracy(self.Y_test, y_pred)
        # print(y_pred)
        # results = generate_results(self.Y_test, y_pred)
        # return results
        #return {"accuracy": np.mean(scores)}