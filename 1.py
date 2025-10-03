import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import fetch_openml

# 1. Load and prepare the Adult Income dataset
print("Loading Adult Income dataset...")

# Fetch the dataset from OpenML
adult = fetch_openml(name='adult', version=2, as_frame=True)
X = adult.data
y = adult.target

# Map target to binary: '>50K' -> 1, '<=50K' -> 0
y = np.where(y == '>50K', 1, 0)

# Convert categorical variables to numerical
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Process categorical features
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column].astype(str))

# Scale numerical features
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 3. Implement a Gaussian Naïve Bayes classifier
print("\nTraining Gaussian Naïve Bayes classifier...")
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# 4. Make predictions
y_pred = nb_classifier.predict(X_test)
y_prob = nb_classifier.predict_proba(X_test)[:, 1]

# 5. Evaluate the model
print("\nModel Evaluation:")
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Precision, Recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# 6. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Gaussian Naive Bayes (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# 7. Perform 10-fold cross-validation
print("\nPerforming 10-fold cross-validation...")
cv_scores = cross_val_score(nb_classifier, X, y, cv=10)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Accuracy Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# 8. Calculate null accuracy (accuracy if we always predict the majority class)
null_accuracy = max(np.mean(y), 1 - np.mean(y))
print(f"\nNull Accuracy: {null_accuracy:.4f}")
print(f"Model improvement over null accuracy: {accuracy - null_accuracy:.4f}")

# 9. Display feature importance (information gain)
# For Naive Bayes, we can look at the variance of each feature for each class
print("\nFeature Importance Analysis:")
feature_importance = pd.DataFrame()
feature_importance['Feature'] = X.columns
feature_importance['Variance_Class0'] = nb_classifier.var_[0]
feature_importance['Variance_Class1'] = nb_classifier.var_[1]
feature_importance['Variance Ratio'] = feature_importance['Variance_Class1'] / feature_importance['Variance_Class0']
feature_importance = feature_importance.sort_values('Variance Ratio', ascending=False)
print(feature_importance.head(10))

# 10. Summary
print("\nModel Summary:")
if accuracy > null_accuracy:
    print(f"The Gaussian Naive Bayes model outperforms the null accuracy by {(accuracy - null_accuracy) * 100:.2f}%")
    print(f"The model correctly classifies {accuracy * 100:.2f}% of the instances.")
else:
    print("The model does not perform better than the null accuracy. Further tuning may be required.")

print("\nInsights:")
print("- Top performing features based on variance ratio:", feature_importance['Feature'].iloc[:3].tolist())
print(f"- Model recall (ability to find all positive samples): {recall:.4f}")
print(f"- Model precision (accuracy of positive predictions): {precision:.4f}")
print(f"- ROC-AUC score: {roc_auc:.4f}")

if roc_auc > 0.7:
    print("The ROC-AUC score suggests the model has good discriminatory power.")
elif roc_auc > 0.5:
    print("The ROC-AUC score suggests the model has some discriminatory power but could be improved.")
else:
    print("The ROC-AUC score suggests the model has poor discriminatory power.")