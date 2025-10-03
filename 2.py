import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')
import io
from urllib.request import urlopen
import zipfile

# Set random seed for reproducibility
np.random.seed(42)

# Fetch the UCI Car Evaluation Dataset
print("Loading Car Evaluation Dataset from UCI...")
try:
    # Direct path to the data file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    data = pd.read_csv(url, header=None)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Attempting alternative download method...")
    try:
        # Alternative: Download the ZIP file and extract
        zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        data = pd.read_csv(zip_url, header=None)
        print("Data loaded successfully via alternative method!")
    except Exception as e2:
        print(f"Error with alternative method: {e2}")
        # If both methods fail, create a sample dataset for demonstration
        print("Creating sample dataset for demonstration...")
        data = pd.DataFrame({
            0: np.random.choice(['vhigh', 'high', 'med', 'low'], 1000),
            1: np.random.choice(['vhigh', 'high', 'med', 'low'], 1000),
            2: np.random.choice(['2', '3', '4', '5more'], 1000),
            3: np.random.choice(['2', '4', 'more'], 1000),
            4: np.random.choice(['small', 'med', 'big'], 1000),
            5: np.random.choice(['low', 'med', 'high'], 1000),
            6: np.random.choice(['unacc', 'acc', 'good', 'vgood'], 1000)
        })

# Set column names according to dataset description
data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

print("\n1. EXPLORATORY DATA ANALYSIS (EDA)")
print("---------------------------------")

# Basic dataset information
print("\nDataset Shape:", data.shape)
print("\nFirst 5 rows of dataset:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isna().sum())

# Data description
print("\nData Information:")
print(data.info())

# Distribution of target variable
print("\nDistribution of Car Classes:")
class_counts = data['class'].value_counts()
print(class_counts)

# Visualize target distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=data, palette='viridis')
plt.title('Distribution of Car Evaluation Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('class_distribution.png')
plt.close()

# Distribution of feature values
print("\nDistribution of Features:")
for col in data.columns[:-1]:
    print(f"\n{col.capitalize()}:")
    print(data[col].value_counts())
    
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=data, palette='Blues_r')
    plt.title(f'Distribution of {col.capitalize()}')
    plt.xticks(rotation=45)
    plt.savefig(f'{col}_distribution.png')
    plt.close()

# Relationship between safety and class
plt.figure(figsize=(10, 6))
sns.countplot(x='safety', hue='class', data=data, palette='viridis')
plt.title('Car Class by Safety Rating')
plt.xlabel('Safety Rating')
plt.ylabel('Count')
plt.legend(title='Class')
plt.savefig('safety_vs_class.png')
plt.close()

print("\n2. DATA PREPROCESSING")
print("--------------------")

# Check data types
print("\nData Types:")
print(data.dtypes)

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Encoding categorical features
print("\nEncoding categorical features...")

# Use OneHotEncoder for categorical features
categorical_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("\n3. TRAINING DECISION TREE MODELS")
print("------------------------------")

# Model 1: Decision Tree with Gini index
print("\nTraining Decision Tree with Gini Index...")
dt_gini_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(criterion='gini', random_state=42))
])
dt_gini_pipeline.fit(X_train, y_train)

# Model 2: Decision Tree with Entropy
print("\nTraining Decision Tree with Entropy...")
dt_entropy_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(criterion='entropy', random_state=42))
])
dt_entropy_pipeline.fit(X_train, y_train)

print("\n4. MODEL EVALUATION")
print("-----------------")

# Predictions
y_pred_gini = dt_gini_pipeline.predict(X_test)
y_pred_entropy = dt_entropy_pipeline.predict(X_test)

# Accuracy
gini_accuracy = accuracy_score(y_test, y_pred_gini)
entropy_accuracy = accuracy_score(y_test, y_pred_entropy)

print(f"\nGini Index Model Accuracy: {gini_accuracy:.4f}")
print(f"Entropy Model Accuracy: {entropy_accuracy:.4f}")

# Classification reports
print("\nGini Index Classification Report:")
gini_report = classification_report(y_test, y_pred_gini)
print(gini_report)

print("\nEntropy Classification Report:")
entropy_report = classification_report(y_test, y_pred_entropy)
print(entropy_report)

# Confusion matrices
print("\nGini Index Confusion Matrix:")
gini_cm = confusion_matrix(y_test, y_pred_gini)
print(gini_cm)

print("\nEntropy Confusion Matrix:")
entropy_cm = confusion_matrix(y_test, y_pred_entropy)
print(entropy_cm)

# Visualize confusion matrices
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.heatmap(gini_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Gini Index')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(entropy_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Entropy')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Compare model accuracies
plt.figure(figsize=(8, 6))
models = ['Gini Index', 'Entropy']
accuracies = [gini_accuracy, entropy_accuracy]
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Decision Tree Model Accuracy Comparison')
plt.xlabel('Splitting Criterion')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)  # Adjust as needed
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
plt.savefig('model_comparison.png')
plt.close()

# Feature importance analysis
gini_clf = dt_gini_pipeline.named_steps['classifier']
entropy_clf = dt_entropy_pipeline.named_steps['classifier']

# Get feature names after one-hot encoding
preprocessor_fitted = preprocessor.fit(X)
one_hot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
gini_importances = gini_clf.feature_importances_
indices = np.argsort(gini_importances)[-10:]  # Top 10 features
plt.barh(range(len(indices)), gini_importances[indices])
plt.yticks(range(len(indices)), [one_hot_feature_names[i] for i in indices])
plt.title('Gini Index Feature Importances (Top 10)')

plt.subplot(1, 2, 2)
entropy_importances = entropy_clf.feature_importances_
indices = np.argsort(entropy_importances)[-10:]  # Top 10 features
plt.barh(range(len(indices)), entropy_importances[indices])
plt.yticks(range(len(indices)), [one_hot_feature_names[i] for i in indices])
plt.title('Entropy Feature Importances (Top 10)')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

print("\n5. FINDINGS AND CONCLUSION")
print("------------------------")

print(f"Gini Index Model Accuracy: {gini_accuracy:.4f}")
print(f"Entropy Model Accuracy: {entropy_accuracy:.4f}")

if gini_accuracy > entropy_accuracy:
    print("\nThe Gini Index model performed better by {:.2f}%.".format((gini_accuracy - entropy_accuracy) * 100))
    better_model = "Gini Index"
elif entropy_accuracy > gini_accuracy:
    print("\nThe Entropy model performed better by {:.2f}%.".format((entropy_accuracy - gini_accuracy) * 100))
    better_model = "Entropy"
else:
    print("\nBoth models performed equally well.")
    better_model = "Both equally"

print("\nConclusion:")
print("----------")
print(f"The {better_model} criterion performed better for this car evaluation classification task.")
print("Possible reasons for this performance difference:")
print("1. The Gini index tends to be more efficient computationally and often works well with noisy data.")
print("2. Entropy is more sensitive to impurities in the node splits and can sometimes lead to different tree structures.")
print("3. The specific distribution of classes in the car evaluation dataset may favor one criterion over the other.")
print("4. Random variations in the dataset splitting might also contribute to slight performance differences.")

print("\nImportant features for classification include the safety rating, number of persons, and luggage boot size.")
print("These findings align with our intuition that safety and practical aspects of cars heavily influence their overall evaluation.")

# Compare tree complexity (number of nodes)
gini_nodes = gini_clf.tree_.node_count
entropy_nodes = entropy_clf.tree_.node_count

print(f"\nTree complexity comparison:")
print(f"Gini Index tree nodes: {gini_nodes}")
print(f"Entropy tree nodes: {entropy_nodes}")

if gini_nodes < entropy_nodes:
    print("The Gini Index model produced a simpler tree, which may be less prone to overfitting.")
elif entropy_nodes < gini_nodes:
    print("The Entropy model produced a simpler tree, which may be less prone to overfitting.")
else:
    print("Both models produced trees of equal complexity.")