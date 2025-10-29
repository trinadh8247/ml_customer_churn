"""
Flask web app for Customer Churn Prediction

This app loads a pre-trained model from model/model.pkl and reproduces
the preprocessing performed in CHURN_PREDICTION.ipynb (imputation, label
encoding, scaling). It does NOT retrain the model — it only fits simple
preprocessors on the training CSV to ensure consistent encodings.

How it works (short):
- On startup: load training data (data/train.csv) to build imputation
  values, label-encoding maps and numeric scaler; load model from
  model/model.pkl
- Expose a form (GET /) to collect feature values
- Accept form POST (POST /predict), apply preprocessing and return
  predicted class and probability.

Notes for maintainers:
- If you re-train the model, save any encoders/scaler used at train time
  and load them here instead of re-fitting on train.csv. That keeps
  predictions stable across retrains.
"""
import os
import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Feature definitions (taken from CHURN_PREDICTION.ipynb)
FEATURE_ORDER = [
    "credit_score",
    "country",
    "gender",
    "age",
    "tenure",
    "acc_balance",
    "prod_count",
    "has_card",
    "is_active",
    "estimated_salary",
]

NUM_COLS = [
    "credit_score",
    "age",
    "tenure",
    "acc_balance",
    "prod_count",
    "estimated_salary",
]

IMPUTE_MEDIAN_COLS = ["credit_score", "acc_balance", "prod_count"]

# Paths
TRAIN_CSV = os.path.join("data", "train.csv")
MODEL_PATH = os.path.join("model", "model.pkl")


def build_preprocessors():
    """Read the train CSV and build simple preprocessors used at inference.

    Important: we do NOT train the ML model here. We only compute
    medians, label encodings, and a StandardScaler fitted on numeric
    columns so inputs are transformed the same way the model expects.
    """
    train = pd.read_csv(TRAIN_CSV)

    # Make sure missing country is represented the same as notebook
    train["country"] = train["country"].fillna("unknown")

    # Median values for simple imputation
    medians = train[IMPUTE_MEDIAN_COLS].median()

    # Fit label encoders on training categories (keeps mapping consistent)
    le_country = LabelEncoder()
    le_gender = LabelEncoder()
    # Fill gender NA if any (not expected) with a placeholder
    train["gender"] = train["gender"].fillna("unknown")
    le_country.fit(train["country"])  # classes_ preserved
    le_gender.fit(train["gender"])    # classes_ preserved

    # Fit scaler for numerical columns
    scaler = StandardScaler()
    # Some numeric columns may have missing values; fill with median before fit
    tmp = train[NUM_COLS].copy()
    for c in IMPUTE_MEDIAN_COLS:
        tmp[c] = tmp[c].fillna(medians[c])
    scaler.fit(tmp)

    # Create mapping dictionaries for safe encoding during inference
    country_map = {v: i for i, v in enumerate(le_country.classes_)}
    gender_map = {v: i for i, v in enumerate(le_gender.classes_)}

    # Keep list of categories for the web form
    countries = sorted(list(train["country"].unique()))
    genders = sorted(list(train["gender"].unique()))

    return {
        "medians": medians,
        "le_country": le_country,
        "le_gender": le_gender,
        "scaler": scaler,
        "country_map": country_map,
        "gender_map": gender_map,
        "countries": countries,
        "genders": genders,
    }


PRE = build_preprocessors()

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place model.pkl there.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def encode_category(value, mapping, default_key="unknown"):
    """Encode a categorical value using mapping dict; fall back to default_key.

    This avoids LabelEncoder throwing on unseen categories and mirrors the
    notebook behaviour where unknown countries were set to 'unknown'.
    """
    if pd.isna(value) or value == "":
        value = default_key
    if value in mapping:
        return mapping[value]
    # fallback to default if present, otherwise first mapping integer
    if default_key in mapping:
        return mapping[default_key]
    return list(mapping.values())[0]


def preprocess_input(row_dict):
    """Take a dict of raw inputs (strings) and return a processed 2D array ready for model."""
    df = pd.DataFrame([row_dict], columns=FEATURE_ORDER)

    # Ensure country/gender NA become 'unknown'
    df["country"] = df["country"].fillna("unknown").astype(str)
    df["gender"] = df["gender"].fillna("unknown").astype(str)

    # Numeric parsing for numerical fields
    for col in NUM_COLS:
        try:
            # parse floats for numeric fields; leave as NaN if parse fails
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = np.nan

    # Binary fields (expected 0/1) - ensure numeric dtype to avoid pandas errors
    binary_cols = ["has_card", "is_active"]
    for b in binary_cols:
        # coerce to numeric and replace NaN (e.g., empty strings) with 0
        df[b] = pd.to_numeric(df[b], errors="coerce").fillna(0).astype(int)

    # Impute medians for specific columns
    for c in IMPUTE_MEDIAN_COLS:
        df[c] = df[c].fillna(PRE["medians"][c])

    # Encode categorical columns using training mapping
    df["country"] = df["country"].apply(lambda v: encode_category(v, PRE["country_map"]))
    df["gender"] = df["gender"].apply(lambda v: encode_category(v, PRE["gender_map"]))

    # Scale numeric columns
    # Make a copy for scaling to avoid warnings
    num_vals = df[NUM_COLS].copy()
    num_scaled = PRE["scaler"].transform(num_vals)
    num_scaled_df = pd.DataFrame(num_scaled, columns=NUM_COLS)

    # Replace numeric columns with scaled versions
    for c in NUM_COLS:
        df[c] = num_scaled_df[c]

    # Ensure final column order matches FEATURE_ORDER
    df = df[FEATURE_ORDER]

    return df


@app.route("/", methods=["GET"])
def index():
    # Provide lists for select fields in the form
    return render_template(
        "index.html",
        countries=PRE["countries"],
        genders=PRE["genders"],
        feature_order=FEATURE_ORDER,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Read form values
    form = request.form
    # Build raw input dict in same order
    raw = {}
    for feat in FEATURE_ORDER:
        raw[feat] = form.get(feat, "")

    X = preprocess_input(raw)

    # Predict
    try:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]
    except Exception as e:
        return f"Prediction failed: {e}", 500

    result = {
        "input": raw,
        "prediction": int(pred),
        "probability": float(proba) if proba is not None else None,
    }

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
