# Utility functions for the student performance prediction project

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    
    Parameters:
    cm (array-like): Confusion matrix
    classes (list): List of class names
    title (str): Title for the plot
    cmap: Colormap for the plot
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def plot_roc_curve(fpr, tpr, title='ROC Curve'):
    """
    This function plots the ROC curve.
    
    Parameters:
    fpr (array-like): False positive rates
    tpr (array-like): True positive rates
    title (str): Title for the plot
    """
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def display_misclassified_samples(X, y_true, y_pred):
    """
    This function displays misclassified samples.
    
    Parameters:
    X (DataFrame): Feature data
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    """
    misclassified_idx = np.where(y_true != y_pred)[0]
    return X.iloc[misclassified_idx]