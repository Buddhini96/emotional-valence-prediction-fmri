from sklearn.model_selection import train_test_split

from data.data_processing_scripts.GNBTrainer import GNBTrainer
from data.data_processing_scripts.LRTrainer import LRTrainer
from data.data_processing_scripts.RFTrainer import RFTrainer
from data.data_processing_scripts.SVMTrainer import SVMTrainer

class ModelTrainer:
    def train_model(self, model_name, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        if model_name == 'linear_regression':
            return LRTrainer(X_train, X_test, Y_train, Y_test).trainmodel()
        elif model_name == 'random_forest':
            return RFTrainer(X_train, X_test, Y_train, Y_test).trainmodel()
        elif model_name == 'svm':
            return SVMTrainer(X_train, X_test, Y_train, Y_test).trainmodel()
        elif model_name == 'GNB':
            return GNBTrainer(X_train, X_test, Y_train, Y_test).trainmodel()
        else:
            raise ValueError("Unsupported model name. Available models: 'linear_regression', 'random_forest', 'svm'")
