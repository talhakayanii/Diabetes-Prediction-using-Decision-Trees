# Diabetes Prediction using Decision Trees
## Description:

This project uses the Diabetes dataset to predict whether a patient has diabetes based on several medical features. The project employs a Decision Tree classifier, optimized using GridSearchCV to improve accuracy, and compares the performance of a basic decision tree versus the optimized version.

## Features:

### Data Preprocessing:

The dataset includes medical features like Glucose, Blood Pressure, Skin Thickness, and BMI.

Zero values in certain columns are treated as missing data and replaced with the median.

The data is split into training and testing sets with 90% of the data used for training and 10% for testing.

### Model Training:

A basic Decision Tree Classifier is trained on the dataset.

Hyperparameter tuning is performed using GridSearchCV for optimization of max_depth, min_samples_split, and min_samples_leaf.

### Model Evaluation:

The model's performance is evaluated using Accuracy, Precision, Recall, and F1-Score.

A comparison of the performance between the basic decision tree and optimized decision tree models is presented.

### Feature Importance:

The most important features influencing the diabetes prediction are displayed based on the feature importance derived from the optimized decision tree.

## Visualization:

Confusion matrices and Feature importance are visualized to provide insights into model performance and relevant features.

## Key Findings:

Basic Model Performance: The basic decision tree model achieves decent accuracy.

Optimized Model Performance: After hyperparameter tuning, the optimized decision tree model outperforms the basic model in both accuracy and F1-score.

Important Features: The most important features for predicting diabetes include Glucose, BMI, and Age.
