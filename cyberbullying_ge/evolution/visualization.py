import matplotlib.pyplot as plt
import seaborn as sns

class ResultVisualizer:
    @staticmethod
    def plot_fitness_curves(fitness_history, title):
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history['avg'], label='Average Fitness')
        plt.fill_between(range(len(fitness_history['avg'])),
                        fitness_history['avg'] - fitness_history['std'],
                        fitness_history['avg'] + fitness_history['std'],
                        alpha=0.2)
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(cm, classes):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.show()