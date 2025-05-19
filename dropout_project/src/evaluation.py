# Evaluation Script for Student Performance Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """
    Evaluate the performance of a model using various metrics.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_prob (array-like): Predicted probabilities.
    model_name (str): Name of the model being evaluated.
    """
    print(f"\n{model_name} Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("AUC:", roc_auc_score(y_true, y_prob))

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot the confusion matrix for the model predictions.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    model_name (str): Name of the model for the title.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_roc_curve(y_true, y_prob, model_name):
    """
    Plot the ROC curve for the model.

    Parameters:
    y_true (array-like): True labels.
    y_prob (array-like): Predicted probabilities.
    model_name (str): Name of the model for the title.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()