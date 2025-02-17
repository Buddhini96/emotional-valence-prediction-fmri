from sklearn.ensemble import RandomForestClassifier
from results_generator import generate_results

class RFTrainer:
    
    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2)
        
    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)
        y_pred = self.model.predict(self.X_test)
        results = generate_results(self.Y_test, y_pred)
        return results