"""
generate_data.py
-----------------
Generates a synthetic heart.csv that follows the exact column schema of the
Kaggle "Heart Disease Dataset" (johnsmith88/heart-disease-dataset), which is
itself based on the UCI Cleveland Heart Disease dataset.

NOTE: If you have internet/Kaggle access, download the real dataset from:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
and replace the generated heart.csv with it. This script exists so the
repository is runnable end-to-end (train -> model.pkl -> Flask API) even
without a Kaggle account/API key.

Columns (14 total, target = presence of heart disease, 1 = disease, 0 = no disease):
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1025  # matches size of the real Kaggle dataset

age = np.random.randint(29, 78, N)
sex = np.random.binomial(1, 0.68, N)  # dataset is skewed male
cp = np.random.randint(0, 4, N)  # chest pain type 0-3
trestbps = np.random.normal(131, 17, N).clip(94, 200).astype(int)  # resting BP
chol = np.random.normal(246, 51, N).clip(126, 564).astype(int)  # cholesterol
fbs = np.random.binomial(1, 0.15, N)  # fasting blood sugar > 120
restecg = np.random.randint(0, 3, N)  # resting ECG results
thalach = np.random.normal(149, 23, N).clip(71, 202).astype(int)  # max heart rate
exang = np.random.binomial(1, 0.33, N)  # exercise induced angina
oldpeak = np.random.exponential(1.0, N).clip(0, 6.2).round(1)  # ST depression
slope = np.random.randint(0, 3, N)  # slope of peak exercise ST segment
ca = np.random.randint(0, 5, N)  # number of major vessels colored by flourosopy
thal = np.random.choice([0, 1, 2, 3], N, p=[0.05, 0.05, 0.55, 0.35])  # thalassemia

# Build target with a logistic relationship to clinically-plausible risk factors
z = (
    -0.6
    + 0.02 * (age - 54)
    + 0.5 * sex
    - 0.35 * (cp == 0).astype(int)
    + 0.012 * (trestbps - 131)
    + 0.004 * (chol - 246)
    - 0.02 * (thalach - 149)
    + 0.5 * exang
    + 0.3 * oldpeak
    + 0.35 * (ca > 0).astype(int)
    + 0.4 * (thal == 3).astype(int)
    + np.random.normal(0, 1.0, N)
)
prob = 1 / (1 + np.exp(-z))
target = np.random.binomial(1, prob)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
    "target": target,
})

df.to_csv("heart.csv", index=False)
print(f"Generated heart.csv with {len(df)} rows.")
print(df["target"].value_counts(normalize=True))
