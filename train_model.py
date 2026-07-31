"""
train_model.py
---------------
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

Loads heart.csv, explores it, splits into train/test, trains a
RandomForestClassifier, evaluates accuracy, and saves the trained
model (+ the feature column order) to model.pkl using joblib.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head(), "\n")

# 3. Identify numerical features and the target variable
target_col = "target"
numerical_features = [c for c in df.columns if c != target_col]
print("Numerical features:", numerical_features)
print("Target variable:", target_col, "\n")

# 4. Check for missing values
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size:  {X_test.shape[0]} rows\n")

# ---------------------------------------------------------------------------
# Task 2: Model Development
# ---------------------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model: RandomForestClassifier")
print(f"Accuracy Score: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model (and the feature order it expects) with Joblib
joblib.dump({"model": model, "features": numerical_features}, "model.pkl")
print("\nSaved trained model to model.pkl")
