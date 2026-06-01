"""
retrain_models.py
─────────────────
Retrains CatBoost, Random Forest, and a Stacking ensemble locally
using the diabetes_model_base.csv dataset and saves compatible pkl files.

Run once (takes ~5-10 min for Random Forest):
  python retrain_models.py
"""

import os, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE      = os.path.join(os.path.dirname(__file__), "models", "clinical_readmission_artifacts")
DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets", "processed", "diabetes_model_base.csv")

RANDOM_STATE = 42

# ── 1. Load features and pre-built preprocessor ───────────────────────────────
print("Loading data and preprocessor ...")
with open(os.path.join(BASE, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

with open(os.path.join(BASE, "preprocessing_pipeline.pkl"), "rb") as f:
    preprocessor = pickle.load(f)

df = pd.read_csv(DATA_PATH, low_memory=False)
TARGET = "readmit_30"

# Cast categoricals to object (required by sklearn)
for col in ["primary_diagnosis_group_reduced", "age_risk_group"]:
    df[col] = df[col].astype(object)

X = df[feature_columns].copy()
y = df[TARGET].copy()
print(f"  X: {X.shape}  Positives: {y.sum():,}/{len(y):,}")

# ── 2. Split (same 80/20 as original notebook) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# ── 3. Preprocess ─────────────────────────────────────────────────────────────
print("\nPreprocessing ...")
X_train_p = preprocessor.transform(X_train)
X_test_p  = preprocessor.transform(X_test)

# ── 4. CatBoost (works directly on raw feature DataFrame) ────────────────────
print("\n[1/3] Training CatBoost ...")
cat_features_idx = [feature_columns.index(c) for c in ["primary_diagnosis_group_reduced", "age_risk_group"]]

cat_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    class_weights=[1, 8],
    verbose=100,
    random_state=RANDOM_STATE
)
cat_model.fit(X_train, y_train, cat_features=cat_features_idx)
p_cat = cat_model.predict_proba(X_test)[:, 1]

with open(os.path.join(BASE, "catboost_model.pkl"), "wb") as f:
    pickle.dump(cat_model, f, protocol=4)
cat_model.save_model(os.path.join(BASE, "catboost_model.cbm"))
print("  CatBoost saved.")

# ── 5. Random Forest ─────────────────────────────────────────────────────────
print("\n[2/3] Training Random Forest (this may take 3-5 min) ...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf_model.fit(X_train_p, y_train)
p_rf = rf_model.predict_proba(X_test_p)[:, 1]

with open(os.path.join(BASE, "random_forest_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f, protocol=4)
print("  Random Forest saved.")

# ── 6. Stacking Ensemble (XGBoost + RF -> LogisticRegression meta) ────────────
print("\n[3/3] Training Stacking Ensemble ...")

from xgboost import XGBClassifier

neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
spw = neg_count / pos_count  # scale_pos_weight for class imbalance

stacking_model = StackingClassifier(
    estimators=[
        ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric="aucpr",
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
        )),
        ("random_forest", RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=10, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE
        ))
    ],
    final_estimator=LogisticRegression(class_weight="balanced", max_iter=500),
    cv=3,
    n_jobs=1,
    passthrough=False
)
stacking_model.fit(X_train_p, y_train)
p_stk = stacking_model.predict_proba(X_test_p)[:, 1]

with open(os.path.join(BASE, "stacking_model.pkl"), "wb") as f:
    pickle.dump(stacking_model, f, protocol=4)
print("  Stacking Ensemble saved.")

# ── 7. Compute and save optimal thresholds ────────────────────────────────────
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.55, 0.005):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)

print("\nComputing optimal thresholds ...")

probs = {"catboost": p_cat, "random_forest": p_rf, "stacking": p_stk}
thresholds = {}

for name, p in probs.items():
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.55, 0.005):
        f1 = f1_score(y_test, (p >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thresholds[name] = float(best_t)
    auc = roc_auc_score(y_test, p)
    print(f"  {name:<16}  threshold={best_t:.4f}  AUC={auc:.4f}  F1={best_f1:.4f}")

with open(os.path.join(BASE, "model_thresholds.pkl"), "wb") as f:
    pickle.dump(thresholds, f, protocol=4)
print("  Thresholds saved.")

print("\nAll done! Run 'python app.py' to start the server.")
