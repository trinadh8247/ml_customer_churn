# Customer Churn Prediction (Web App)

This repository contains a customer churn prediction project and a simple
Flask web application that uses a pre-trained model to predict whether a
customer will exit (churn).

Key points
- Problem: Predict customer churn/exit status using customer attributes.
- Dataset: training and test CSVs are in `data/` (source: provided dataset used in the notebook).
- Best model (from experiments in `CHURN_PREDICTION.ipynb`): LightGBM with
	F1 = 0.60 and Accuracy = 0.86.
- Model artifact: `model/model.pkl` (must be present for the web app to run).

What I added
- A small Flask app (`app.py`) that loads `model/model.pkl` and exposes a
	web form for making single predictions.
- HTML templates in `templates/` (Bootstrap-based UI) to collect features and
	show results.
- `requirements.txt` with Python dependencies.

Files required to run
- `model/model.pkl` — the pre-trained model file produced from training (place it in `model/`).
- `data/train.csv` — used by the app to build encoders, medians and scaler so
	feature preprocessing matches the notebook. (The app does not retrain the model; it only fits preprocessing objects on train.csv.)
- `CHURN_PREDICTION.ipynb` — reference notebook containing EDA, preprocessing
	and modeling steps.

Quick setup
1. Create and activate a Python virtual environment (recommended):

	 # Windows PowerShell
	 python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies:

	 pip install -r requirements.txt

3. Ensure `model/model.pkl` and `data/train.csv` exist in the repository.

4. Run the app:

	 python app.py

	 The app will be available at http://127.0.0.1:5000/ by default.

Usage
1. Open the app in your browser.
2. Fill the form with customer attributes: credit score, country, gender,
	 age, tenure, account balance, product count, has_card (0/1), is_active (0/1),
	 estimated salary.
3. Click Predict to get a churn (exit_status) prediction and model probability.

Preprocessing details (matching notebook)
- Missing countries in train were replaced with the string `unknown`.
- Numeric columns `credit_score`, `acc_balance`, `prod_count` were imputed
	using median values computed from the training CSV.
- Numerical columns are StandardScaled using a scaler fit on training numeric features.
- Categorical columns (`country`, `gender`) were label-encoded using the
	training categories so inference is consistent with training.

Notes and limitations
- The app does not retrain the model. If you re-train, save encoders and the
	scaler used at training time and update the app to load them for perfect
	reproducibility.
- If the app receives a country/gender value unseen during training it will
	map it to the `unknown` category where possible.

Project technologies
- Python, Flask
- pandas, numpy, scikit-learn
- LightGBM (trained model)
- Bootstrap for UI

Results & insights
- Best performing model found: LightGBM (F1=0.60, Accuracy=0.86). This
	indicates a reasonably strong classifier for churn with balanced precision/recall.
- Typical high-impact features: credit score, account balance and country —
	these can be used by product and retention teams to prioritize interventions.

Contributor guidelines
- This is provided as a personal/project repo. If you want to accept external
	contributions, add a `CONTRIBUTING.md` with your contribution rules. For small
	tweaks (UI, docs), feel free to open a PR.

Contact / Next steps
- To improve production readiness: persist the preprocessing objects used at
	training (scaler, encoders) into files and load them in the app. Add tests
	and basic input validation on the server side. Containerize with Docker if
	shipping to a cloud environment.

---
Updated: see `app.py` and `templates/` for the web UI implementation.

