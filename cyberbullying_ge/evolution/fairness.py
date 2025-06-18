class FairnessValidator:
    def __init__(self, demographic_groups):
        self.groups = demographic_groups
        
    def calculate_disparity(self, y_true, y_pred, metadata):
        disparities = []
        for group in self.groups:
            group_acc = []
            for value in metadata[group].unique():
                mask = metadata[group] == value
                acc = accuracy_score(y_true[mask], y_pred[mask])
                group_acc.append(acc)
            disparities.append(max(group_acc) - min(group_acc))
        return max(disparities)