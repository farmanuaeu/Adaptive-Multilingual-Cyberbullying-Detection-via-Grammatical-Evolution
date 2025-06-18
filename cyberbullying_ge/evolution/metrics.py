from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }