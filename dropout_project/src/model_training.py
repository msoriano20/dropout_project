# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

def load_data():
    # Load the datasets
    mat_data = pd.read_csv('../data/student-mat.csv', sep=';')
    por_data = pd.read_csv('../data/student-por.csv', sep=';')
    return mat_data, por_data

def train_logistic_regression(X_train, y_train):
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    return logreg

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_decision_tree(X_train, y_train):
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)
    return dtree

def hyperparameter_tuning(X_train, y_train):
    # Hyperparameter tuning for Decision Tree
    param_grid_tree = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5, scoring='f1')
    grid_tree.fit(X_train, y_train)
    
    # Hyperparameter tuning for Logistic Regression
    param_grid_logreg = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    grid_logreg = GridSearchCV(logreg, param_grid_logreg, cv=5, scoring='f1')
    grid_logreg.fit(X_train, y_train)

    return grid_tree.best_estimator_, grid_logreg.best_estimator_

def main():
    mat_data, por_data = load_data()
    
    # Assuming the target variable is 'G3' and features are preprocessed
    X = mat_data.drop('G3', axis=1)
    y = (mat_data['G3'] >= 10).astype(int)  # Binary target: 1 if passed, 0 if not
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    logreg_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    dtree_model = train_decision_tree(X_train, y_train)

    # Hyperparameter tuning
    best_tree, best_logreg = hyperparameter_tuning(X_train, y_train)

    # Evaluate models (this part can be moved to evaluation.py)
    y_pred_logreg = logreg_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_tree = best_tree.predict(X_test)

    print("Logistic Regression F1 Score:", f1_score(y_test, y_pred_logreg))
    print("Random Forest F1 Score:", f1_score(y_test, y_pred_rf))
    print("Best Decision Tree F1 Score:", f1_score(y_test, y_pred_tree))

if __name__ == "__main__":
    main()