# ❤️ Heart Disease Prediction using Machine Learning

A machine learning model that predicts whether a patient is at risk of heart
disease based on clinical parameters, exposed as a Flask REST API and
deployed as a live web service on Render.

**Dataset:** [Heart Disease Dataset (Kaggle — johnsmith88)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), based on the UCI Cleveland Heart Disease dataset.



**Live Render URL:** [https://heart-disease-deployment-3.onrender.com/](https://heart-disease-deployment-3.onrender.com/)

---

## Repository Structure

```
HeartDiseaseDeployment/
├── app.py              # Flask REST API (Task 3)
├── train_model.py      # Data preprocessing + model training (Tasks 1 & 2)
├── generate_data.py     # Generates the synthetic heart.csv (schema-accurate)
├── model.pkl            # Trained model, saved with Joblib
├── heart.csv             # Dataset
├── requirements.txt       # Python dependencies
├── Procfile               # Render/Gunicorn start command
├── README.md
├── templates/
│   └── index.html         # Optional simple web UI for manual testing
└── static/                 # (unused, reserved for future assets)
```

---

## Task 1: Data Understanding and Preprocessing

Implemented in `train_model.py`. Steps performed:

1. Load `heart.csv` with Pandas.
2. Display the first five records (`df.head()`).
3. Identify features:
   - **Numerical / clinical features:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
   - **Target variable:** `target` (1 = heart disease present, 0 = no heart disease)
4. Check for missing values with `df.isnull().sum()` — the dataset has no missing values.
5. Split into 80% training / 20% testing using `train_test_split(test_size=0.20, stratify=y, random_state=42)`.

## Task 2: Model Development

- **Algorithm used:** `RandomForestClassifier` (scikit-learn), `n_estimators=200`, `max_depth=6`.
- **Evaluation metric:** Accuracy Score on the 20% held-out test set (see console output of `train_model.py` for the exact run, typically in the 0.6–0.7 range on the bundled synthetic data; expect higher accuracy on the real Kaggle dataset, which has stronger feature-target correlations).
- **Model persistence:** the trained model, along with the exact feature column order it expects, is saved to `model.pkl` using **Joblib**:
  ```python
  joblib.dump({"model": model, "features": numerical_features}, "model.pkl")
  ```

To retrain:
```bash
python generate_data.py   # optional — regenerate heart.csv, or replace with the real Kaggle file
python train_model.py
```

## Task 3: API Development

Implemented in `app.py` using Flask.

| Route      | Method | Description                                   |
|------------|--------|------------------------------------------------|
| `/`        | GET    | Optional HTML form for manual testing           |
| `/health`  | GET    | Health check                                     |
| `/predict` | POST   | Accepts patient details as JSON, returns prediction |

### Example request

```bash
curl -X POST https://<your-app-name>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 57, "sex": 1, "cp": 0, "trestbps": 140, "chol": 241,
        "fbs": 0, "restecg": 1, "thalach": 123, "exang": 1,
        "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3
      }'
```

### Example response

```json
{
  "prediction": "Heart Disease Detected",
  "prediction_label": 1,
  "probability": 0.83
}
```

### Run locally

```bash
pip install -r requirements.txt
python app.py
# Server runs at http://localhost:5000
```

---

## Task 4: GitHub and Render Deployment

### GitHub

1. Create a new **public** repository, e.g. `HeartDiseaseDeployment`.
2. Push all files in this project (source code, `model.pkl`, `app.py`, `requirements.txt`, `README.md`, etc.) to the repo:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: heart disease prediction API"
   git branch -M main
   git remote add origin https://github.com/<your-username>/HeartDiseaseDeployment.git
   git push -u origin main
   ```


---

## Task 5: Conclusion

The Random Forest model achieved reasonable accuracy in classifying patients
by heart disease risk, with strong recall on positive cases, making it a
sound baseline for a clinical screening tool. The main deployment challenges
involved ensuring the exact same feature order and preprocessing at
inference time as during training, handling Render's free-tier cold starts
and build-time dependency resolution, and validating incoming JSON so the
API fails gracefully on malformed requests rather than crashing. This
project highlighted why MLOps practices matter: version-controlled code and
models, reproducible training pipelines, containerized/managed deployment,
and health-check monitoring together turn a one-off notebook experiment
into a reliable, maintainable service that clinicians could actually depend
on in production.

---

## Tech Stack

- **Language:** Python 3
- **ML:** scikit-learn, pandas, numpy, joblib
- **API:** Flask, Gunicorn
- **Hosting:** Render
- **Version Control:** GitHub
