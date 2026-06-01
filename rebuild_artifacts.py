"""
rebuild_artifacts.py
────────────────────
Rebuilds the preprocessing_pipeline.pkl using the local sklearn version
and re-saves the CatBoost, Random Forest, and Stacking model weights into
fresh pkl files that are compatible with the current Python/sklearn environment.

Run once:
  python rebuild_artifacts.py
"""

import pickle, warnings, os, sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE        = os.path.join(os.path.dirname(__file__), "models", "clinical_readmission_artifacts")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "datasets", "processed", "diabetes_model_base.csv")

# ── 1. Load the feature column list (this pkl is tiny and loads fine) ─────────
print("Loading feature_columns.pkl …")
with open(os.path.join(BASE, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)
print(f"  {len(feature_columns)} features: {feature_columns[:5]} …")

# ── 2. Load the training data to fit the preprocessor ─────────────────────────
print("\nLoading diabetes_model_base.csv …")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Shape: {df.shape}")

TARGET = "readmit_30"
if TARGET not in df.columns:
    sys.exit(f"ERROR: '{TARGET}' column not found in {DATA_PATH}")

X = df[feature_columns].copy()
y = df[TARGET].copy()
print(f"  X shape: {X.shape}  |  y positives: {y.sum():,} / {len(y):,}")

# ── 3. Define and fit the preprocessor (same spec as notebook 05) ─────────────
CATEGORICAL_COLS = ["primary_diagnosis_group_reduced", "age_risk_group"]
NUMERIC_COLS     = [c for c in feature_columns if c not in CATEGORICAL_COLS]

for col in CATEGORICAL_COLS:
    X[col] = X[col].astype(object)

print("\nFitting ColumnTransformer (OneHotEncoder + StandardScaler) …")
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ("num", StandardScaler(), NUMERIC_COLS),
    ],
    remainder="drop",
)
preprocessor.fit(X)
print("  Done.")

# ── 4. Save the new preprocessor ──────────────────────────────────────────────
out_pipeline = os.path.join(BASE, "preprocessing_pipeline.pkl")
with open(out_pipeline, "wb") as f:
    pickle.dump(preprocessor, f)
print(f"\n✅ Saved rebuilt preprocessor → {out_pipeline}")

# ── 5. Re-save CatBoost using its native save/load (bypasses pickle issues) ──
print("\nLoading and re-saving CatBoost model …")
cat_path     = os.path.join(BASE, "catboost_model.pkl")
cat_cbm_path = os.path.join(BASE, "catboost_model.cbm")   # native format

try:
    with open(cat_path, "rb") as f:
        cat_model = pickle.load(f)
    print("  Loaded via pickle (same-machine – should work).")
except Exception as e:
    print(f"  pickle load failed ({e}); trying CatBoost native format …")
    cat_model = CatBoostClassifier()
    if os.path.exists(cat_cbm_path):
        cat_model.load_model(cat_cbm_path)
        print("  Loaded via .cbm native format.")
    else:
        sys.exit("ERROR: Cannot load CatBoost model. No .cbm file found either.")

# Save native .cbm for robustness
cat_model.save_model(cat_cbm_path)
# Also re-dump with current pickle protocol
with open(cat_path, "wb") as f:
    pickle.dump(cat_model, f, protocol=4)
print(f"  ✅ CatBoost re-saved → {cat_path}  +  {cat_cbm_path}")

# ── 6. Re-save Random Forest ──────────────────────────────────────────────────
print("\nLoading and re-saving Random Forest model …")
rf_path = os.path.join(BASE, "random_forest_model.pkl")
try:
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f, protocol=4)
    print(f"  ✅ Random Forest re-saved → {rf_path}")
except Exception as e:
    print(f"  ⚠  Could not load Random Forest: {e}")
    print("     The app will fall back to CatBoost only for RF predictions.")

# ── 7. Re-save Stacking model ─────────────────────────────────────────────────
print("\nLoading and re-saving Stacking model …")
stk_path = os.path.join(BASE, "stacking_model.pkl")
try:
    with open(stk_path, "rb") as f:
        stk_model = pickle.load(f)
    with open(stk_path, "wb") as f:
        pickle.dump(stk_model, f, protocol=4)
    print(f"  ✅ Stacking model re-saved → {stk_path}")
except Exception as e:
    print(f"  ⚠  Could not load Stacking model: {e}")
    print("     Run notebook 06 locally to retrain the stacking layer.")

# ── 8. Verify round-trip ──────────────────────────────────────────────────────
print("\n── Verification round-trip ──────────────────────────────")
X_proc = preprocessor.transform(X.iloc[:5])

try:
    p_cat = cat_model.predict_proba(X_proc)[:, 1]
    print(f"  CatBoost  proba sample: {p_cat.round(4)}")
except Exception as e:
    print(f"  CatBoost predict error: {e}")

try:
    with open(rf_path, "rb") as f:
        rf2 = pickle.load(f)
    p_rf = rf2.predict_proba(X_proc)[:, 1]
    print(f"  RandomForest proba sample: {p_rf.round(4)}")
except Exception as e:
    print(f"  RF predict error: {e}")

try:
    with open(stk_path, "rb") as f:
        stk2 = pickle.load(f)
    p_stk = stk2.predict_proba(X_proc)[:, 1]
    print(f"  Stacking proba sample: {p_stk.round(4)}")
except Exception as e:
    print(f"  Stacking predict error: {e}")

print("\n✅ rebuild_artifacts.py complete.")
