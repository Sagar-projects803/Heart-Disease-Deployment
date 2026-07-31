"""
app.py
------
Task 3: API Development

Flask REST API that loads the trained model (model.pkl) and exposes:
  GET  /            -> simple HTML form (optional UI) to test predictions
  GET  /health      -> health check endpoint
  POST /predict     -> accepts patient details as JSON, returns a JSON prediction
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model bundle (model + expected feature order) once at startup
MODEL_BUNDLE = joblib.load("model.pkl")
model = MODEL_BUNDLE["model"]
FEATURES = MODEL_BUNDLE["features"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient details as JSON, e.g.:
    {
        "age": 57, "sex": 1, "cp": 0, "trestbps": 140, "chol": 241,
        "fbs": 0, "restecg": 1, "thalach": 123, "exang": 1,
        "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3
    }
    Returns:
    { "prediction": "Heart Disease Detected", "prediction_label": 1, "probability": 0.83 }
    """
    try:
        data = request.get_json(force=True)

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required field(s): {', '.join(missing)}"
            }), 400

        # Build a single-row DataFrame in the exact feature order used at training time
        input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][prediction])

        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result_text,
            "prediction_label": prediction,
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # For local development only. Render will use gunicorn (see Procfile / start command).
    app.run(host="0.0.0.0", port=5000, debug=True)
