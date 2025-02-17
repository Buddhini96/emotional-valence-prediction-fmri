from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def generate_results(y_true, y_pred):
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
            }
    
    