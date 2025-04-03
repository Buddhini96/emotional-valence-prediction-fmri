from sklearn.neighbors import KNeighborsClassifier
from results_generator import generate_results

class KNNClassifier:
    
    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = KNeighborsClassifier(n_neighbors=10)

        
    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)
        y_pred = self.model.predict(self.X_test)
        results = generate_results(self.Y_test, y_pred)
        return results