# Student Performance Prediction Project

This project aims to predict student performance based on various demographic, academic, and social factors. The analysis is conducted using machine learning techniques, and the project is structured to facilitate easy replication of the analysis pipeline.

## Project Structure

```
student-performance-prediction
├── data
│   ├── student-mat.csv         # Student performance data for mathematics
│   └── student-por.csv         # Student performance data for Portuguese
├── notebooks
│   └── sprint_2_and_3.ipynb    # Jupyter notebook containing the analysis pipeline
├── src
│   ├── data_preprocessing.py    # Script for data loading and preprocessing
│   ├── feature_engineering.py    # Script for feature engineering
│   ├── model_training.py         # Script for training machine learning models
│   ├── evaluation.py             # Script for model evaluation
│   └── utils.py                  # Utility functions used across modules
├── requirements.txt              # List of required Python packages
└── README.md                     # Documentation for the project
```

## Getting Started

To replicate the analysis pipeline, follow these steps:

### Prerequisites

Ensure you have Python installed on your machine. It is recommended to use a virtual environment to manage dependencies.

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/msoriano20/dropout_project.git
   cd student-performance-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Analysis

1. Open the Jupyter notebook:
   ```
   jupyter notebook notebooks/predict_dropout.ipynb
   ```

2. Follow the instructions in the notebook to execute the analysis pipeline. The notebook includes sections for:
   - Data loading
   - Data preprocessing
   - Feature engineering
   - Model training
   - Model evaluation and hyperparameter tuning

### Scripts Overview

- **data_preprocessing**: Handles data loading, missing value treatment, and feature scaling.
- **feature_engineering**: Creates new features to enhance model performance.
- **model_training**: Trains various models and performs hyperparameter tuning.
- **evaluation**: Evaluates model performance using various metrics and visualizations.
- **utils**: Contains utility functions for plotting and displaying results.

## Conclusion

This project provides a comprehensive analysis of student performance prediction using machine learning techniques. By following the instructions above, you can replicate the analysis and explore the impact of different features on student outcomes.