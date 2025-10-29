# test_predict.py — quick smoke test for the Flask app's preprocessing + model
import traceback

from app import preprocess_input, model

sample = {
    "credit_score": "650",
    "country": "Germany",
    "gender": "Male",
    "age": "45",
    "tenure": "3",
    "acc_balance": "15000",
    "prod_count": "2",
    "has_card": "1",
    "is_active": "1",
    "estimated_salary": "70000"
}

try:
    X = preprocess_input(sample)
    print("Preprocessed features (DataFrame):")
    print(X.to_string(index=False))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][1]
    else:
        proba = None
    pred = model.predict(X)[0]
    print(f"Prediction: {int(pred)}")
    print(f"Probability (churn): {proba}")
except Exception as e:
    print("Smoke test failed:")
    traceback.print_exc()
