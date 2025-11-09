# =======================================
# Final Model Pipeline (Deployment)
# =======================================

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# -------------------------------
# Load Data
# -------------------------------
data = pd.read_csv('data/processed/balanced_data.csv')

# -------------------------------
# Feature Exclusion
# -------------------------------
exclude_features = [
    "num_killed","extended_event","num_killed_us",
    "num_wounded","num_wounded_us","num_wounded_terrorists",
    "property_damage","property_extent_code","property_extent",
    "property_value","hostage_incident","ransom_demanded",
    "hostage_outcome_code","hostage_outcome","num_released",
    "claim_mode","secondary_weapon_subtype","city","vicinity_area",
    "successful_attack", "attack_claimed"
]

# Separate features and target
y = data["successful_attack"]
X = data.drop(columns=exclude_features + ["successful_attack"], errors='ignore')

# -------------------------------
# Preprocessing
# -------------------------------
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# -------------------------------
# XGBoost Model (Tuned)
# -------------------------------
xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    gamma=0.2,
    reg_alpha=0.4,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# -------------------------------
# Pipeline
# -------------------------------
model = Pipeline([
    ("preprocess", preprocess),
    ("xgb", xgb_model)
])

# -------------------------------
# Train Final Model on All Data
# -------------------------------
print("Training final model on full dataset...")
model.fit(X, y)
print("Training complete!")

# -------------------------------
# Save Model for Deployment
# -------------------------------
joblib.dump(model, "final_model.pkl")
print("Model saved successfully as 'final_model.pkl'")