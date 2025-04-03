from sklearn.linear_model import LogisticRegression
from python_scripts.utils.results_generator import calculate_accuracy


class LRTrainer:
    
    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = LogisticRegression(solver='liblinear', C=0.007, penalty='l2', max_iter=1000, tol=0.0005, class_weight='balanced')
        
    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)
        y_pred = self.model.predict(self.X_test)
        return calculate_accuracy(self.Y_test, y_pred)
        # results = generate_results(self.Y_test, y_pred)
        # return results