import pandas as pd
import numpy as np
from ibm_watson_studio_lib import access_project_or_space
# Initialize Watson Studio library with your project token
# Replace <PROJECT_TOKEN> with the long token you copied earlier
wslib = access_project_or_space({"token": "<PROJECT_TOKEN>"})
# Download the dataset file to the notebook environment
wslib.download_file("creditcard.csv")
print("Dataset downloaded successfully!")

# Load CSV into pandas DataFrame
df = pd.read_csv("creditcard.csv")
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
df.head()

# Dataset shape and structure
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes.value_counts()}")
# Check for missing values
missing_count = df.isnull().sum().sum()
print(f"\nTotal missing values: {missing_count}")
if missing_count == 0:
    print("✓ Clean dataset - no missing values!")

# Class distribution - THE KEY CHALLENGE
print("Class Distribution:")
print(df['Class'].value_counts())
print("\nPercentages:")
print(df['Class'].value_counts(normalize=True) * 100)
# Calculate fraud percentage
fraud_count = df['Class'].sum()
total_count = len(df)
fraud_ratio = (fraud_count / total_count) * 100
print(f"\n￿ Fraud Rate: {fraud_ratio:.3f}%")
print(f"This means only {fraud_count} out of {total_count:,}transactions are fraud!")
print(f"Imbalance ratio: {total_count / fraud_count:.1f}:1")

# Statistical summary
print("Feature Statistics:")
df.describe()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Separate features (X) from target (y)
X = df.drop(columns='Class')
y = df['Class']
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# StandardScaler: mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features normalized!")
print(f"Scaled features shape: {X_scaled.shape}")

# Split with STRATIFICATION (critical for imbalanced data!)
X_train, X_test, y_train, y_test = train_test_split(
X_scaled,
y,
test_size=0.2,
random_state=42,
stratify=y # Maintains fraud ratio in both sets
)
print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
# Verify fraud rate preserved
train_fraud_rate = y_train.mean() * 100
test_fraud_rate = y_test.mean() * 100
print(f"\nTrain fraud rate: {train_fraud_rate:.3f}%")
print(f"Test fraud rate: {test_fraud_rate:.3f}%")

from sklearn.linear_model import LogisticRegression
import time
# Initialize Logistic Regression
clf = LogisticRegression(
max_iter=1000, # Ensure convergence
random_state=42
)
# Train and time
print("Training Logistic Regression model...")
start_time = time.time()
clf.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"✓ Model trained in {training_time:.2f} seconds")

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# Make predictions
y_pred = clf.predict(X_test)
# Overall accuracy (will be misleading!)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("￿￿ High accuracy doesn't mean good fraud detection!\n")
# Detailed metrics per class
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate(0)', 'Fraud (1)']))

import matplotlib.pyplot as plt
import seaborn as sns
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nConfusion Matrix Values:")
print(f"True Negatives (TN): {cm[0,0]:,}")
print(f"False Positives (FP): {cm[0,1]:,} ← Legitimate flagged as fraud")
print(f"False Negatives (FN): {cm[1,0]:,} ← Fraud missed! ￿")
print(f"True Positives (TP): {cm[1,1]:,} ← Fraud caught!")
# Visualize (with Watson Studio workaround for label bug)
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, cmap='Purples', cbar=False, linewidths=0.5)
# Force tick labels (Watson bug workaround)
ax.set_xticklabels(['Legitimate', 'Fraud'])
ax.set_yticklabels(['Legitimate', 'Fraud'], rotation=0)
# Manually annotate cells (guaranteed visible)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j + 0.5, i + 0.5, f"{cm[i, j]:,}",ha="center", va="center",color="black", fontsize=14, fontweight='bold',zorder=3, transform=ax.transData)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Fraud Detection - Confusion Matrix', fontsize=14,
fontweight='bold')
plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestClassifier
# Train Random Forest with balanced class weights
# class_weight='balanced' automatically adjusts for imbalance
rf = RandomForestClassifier(
class_weight='balanced', # Penalize misclassifying minority class
n_estimators=100,
max_depth=20,
random_state=42,
n_jobs=-1, # Use all CPU cores
verbose=1
)
print("Training Random Forest (this may take 1-2 minutes)...")
start_time = time.time()
rf.fit(X_train, y_train)
rf_training_time = time.time() - start_time
print(f"✓ Random Forest trained in {rf_training_time:.2f} seconds")


# Predict and evaluate
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate(0)', 'Fraud (1)']))
# Compare to Logistic Regression
from sklearn.metrics import f1_score
lr_f1 = f1_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred_rf)
print(f"\nModel Comparison (Fraud Class F1-Score):")
print(f"Logistic Regression: {lr_f1:.4f}")
print(f"Random Forest: {rf_f1:.4f}")
print(f"\nImprovement: {((rf_f1 - lr_f1) / lr_f1 * 100):.1f}%")