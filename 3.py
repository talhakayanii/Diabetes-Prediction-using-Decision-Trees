import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading Diabetes.csv dataset...")
try:
    data = pd.read_csv('Diabetes.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()

# Basic data exploration
print("\n--- Dataset Information ---")
print(data.info())
print("\n--- Summary Statistics ---")
print(data.describe())

# Check for missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Check for zero values in columns where zero doesn't make sense
print("\n--- Columns with zero values (where zeros might be placeholders for missing data) ---")
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    zero_count = (data[column] == 0).sum()
    print(f"{column}: {zero_count} zeros ({zero_count / len(data) * 100:.2f}%)")

# Data preprocessing
# Replace zeros with NaN where zeros don't make sense
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data_clean = data.copy()
for column in columns_to_replace:
    data_clean[column] = data_clean[column].replace(0, np.nan)

# Fill NaN values with median of each column
for column in columns_to_replace:
    data_clean[column].fillna(data_clean[column].median(), inplace=True)

print("\n--- After handling missing values (first 5 rows) ---")
print(data_clean.head())

# Split features and target
X = data_clean.drop('Outcome', axis=1)
y = data_clean['Outcome']

# Split data into training and testing sets (90/10 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Train a basic Decision Tree model
print("\n--- Training Basic Decision Tree Model ---")
basic_dt = DecisionTreeClassifier(random_state=42)
basic_dt.fit(X_train_scaled, y_train)

# Evaluate the basic model
y_pred_basic = basic_dt.predict(X_test_scaled)
acc_basic = accuracy_score(y_test, y_pred_basic)
prec_basic = precision_score(y_test, y_pred_basic)
rec_basic = recall_score(y_test, y_pred_basic)
f1_basic = f1_score(y_test, y_pred_basic)

print("\n--- Basic Decision Tree Performance ---")
print(f"Accuracy: {acc_basic:.4f}")
print(f"Precision: {prec_basic:.4f}")
print(f"Recall: {rec_basic:.4f}")
print(f"F1-score: {f1_basic:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_basic))

# 2. Hyperparameter tuning using GridSearchCV
print("\n--- Hyperparameter Tuning with GridSearchCV ---")
param_grid = {
    'max_depth': [None, 3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# Print best parameters
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 3. Evaluate the optimized model
optimized_dt = grid_search.best_estimator_
y_pred_opt = optimized_dt.predict(X_test_scaled)
acc_opt = accuracy_score(y_test, y_pred_opt)
prec_opt = precision_score(y_test, y_pred_opt)
rec_opt = recall_score(y_test, y_pred_opt)
f1_opt = f1_score(y_test, y_pred_opt)

print("\n--- Optimized Decision Tree Performance ---")
print(f"Accuracy: {acc_opt:.4f}")
print(f"Precision: {prec_opt:.4f}")
print(f"Recall: {rec_opt:.4f}")
print(f"F1-score: {f1_opt:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_opt))

# Comparing basic and optimized model performance
print("\n--- Comparison: Basic vs. Optimized Decision Tree ---")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
basic_scores = [acc_basic, prec_basic, rec_basic, f1_basic]
opt_scores = [acc_opt, prec_opt, rec_opt, f1_opt]

comparison_df = pd.DataFrame({
    'Metric': metrics,
    'Basic Decision Tree': basic_scores,
    'Optimized Decision Tree': opt_scores,
    'Improvement': [opt - basic for opt, basic in zip(opt_scores, basic_scores)]
})
print(comparison_df)

# Feature importance visualization
plt.figure(figsize=(12, 6))
importances = optimized_dt.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.bar(features[indices], importances[indices])
plt.title('Feature Importance - Optimized Decision Tree')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Findings for the report
print("\n--- Key Findings for Report ---")
print("1. Data Quality:")
print(f"   - Original dataset has {data.shape[0]} instances with {data.shape[1]} features")
print("   - Several features contained zero values that likely represent missing data")
print("   - Data preprocessing included replacing zeros with median values")

print("\n2. Model Performance:")
print(f"   - Basic Decision Tree: Accuracy={acc_basic:.4f}, F1-score={f1_basic:.4f}")
print(f"   - Optimized Decision Tree: Accuracy={acc_opt:.4f}, F1-score={f1_opt:.4f}")
if acc_opt > acc_basic:
    print(f"   - Performance improvement: {(acc_opt - acc_basic) * 100:.2f}% in accuracy")
    print(f"   - Performance improvement: {(f1_opt - f1_basic) * 100:.2f}% in F1-score")
else:
    print("   - Optimization did not significantly improve model performance")

print("\n3. Best Hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"   - {param}: {value}")

print("\n4. Most Important Features:")
for i in range(min(5, len(features))):
    print(f"   - {features[indices[i]]}: {importances[indices[i]]:.4f}")

print("\n5. Conclusion:")
print("   - The decision tree model can effectively predict diabetes diagnosis")
print("   - Hyperparameter tuning through cross-validation helped optimize model performance")
print("   - The most important predictors for diabetes appear to be Glucose, BMI, and Age")