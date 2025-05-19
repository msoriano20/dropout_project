# Contents of /student-performance-prediction/student-performance-prediction/src/feature_engineering.py

import pandas as pd

def create_features(df):
    """
    Create new features for the student performance dataset.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing student data.

    Returns:
    pd.DataFrame: The DataFrame with new features added.
    """
    # Create average_grade feature
    df['average_grade'] = (df['G1'] + df['G2']) / 2

    # Create high_absentee binary feature
    df['high_absentee'] = (df['absences'] > 10).astype(int)

    return df

def select_features(df):
    """
    Select relevant features for modeling.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing student data.

    Returns:
    pd.DataFrame: The DataFrame with selected features.
    """
    selected_features = [
        'sex_F', 'sex_M', 'age', 'studytime', 'failures', 'schoolsup_yes',
        'average_grade', 'absences', 'high_absentee'
    ]
    
    return df[selected_features]