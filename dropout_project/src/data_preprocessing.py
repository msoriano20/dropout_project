# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(mat_file_path, por_file_path):
    """
    Load the student performance datasets from the specified file paths.

    Parameters:
    mat_file_path (str): Path to the mathematics dataset.
    por_file_path (str): Path to the Portuguese dataset.

    Returns:
    tuple: A tuple containing two DataFrames (mathematics and Portuguese).
    """
    mat_data = pd.read_csv(mat_file_path, sep=';')
    por_data = pd.read_csv(por_file_path, sep=';')
    return mat_data, por_data

def preprocess_data(df):
    """
    Preprocess the student performance DataFrame by handling missing values,
    encoding categorical variables, and scaling numerical features.

    Parameters:
    df (DataFrame): The DataFrame containing student performance data.

    Returns:
    DataFrame: The preprocessed DataFrame ready for analysis.
    """
    # Check for missing values
    if df.isnull().sum().any():
        df = df.fillna(method='ffill')  # Forward fill for missing values

    # Convert categorical columns to numeric using one-hot encoding
    df = pd.get_dummies(df)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df