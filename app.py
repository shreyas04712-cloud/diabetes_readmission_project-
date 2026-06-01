import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

app = Flask(__name__)

# -- Load models and artefacts -----------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "models", "clinical_readmission_artifacts")

with open(os.path.join(BASE, "preprocessing_pipeline.pkl"), "rb") as f:
    pipeline = pickle.load(f)

# CatBoost: try pickle first, fall back to native .cbm format
try:
    with open(os.path.join(BASE, "catboost_model.pkl"), "rb") as f:
        catboost_model = pickle.load(f)
except Exception:
    from catboost import CatBoostClassifier as _CatBoost
    catboost_model = _CatBoost()
    catboost_model.load_model(os.path.join(BASE, "catboost_model.cbm"))

with open(os.path.join(BASE, "random_forest_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(os.path.join(BASE, "stacking_model.pkl"), "rb") as f:
    stacking_model = pickle.load(f)

with open(os.path.join(BASE, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

with open(os.path.join(BASE, "model_thresholds.pkl"), "rb") as f:
    thresholds = pickle.load(f)

CAT_FEATURE_COLS = ["primary_diagnosis_group_reduced", "age_risk_group"]

print("[OK] All models and artefacts loaded.")
print("Thresholds -> CatBoost:", round(float(thresholds['catboost']), 4),
      "| RF:", round(float(thresholds['random_forest']), 4),
      "| Stacking:", round(float(thresholds['stacking']), 4))

# ── Feature engineering (ported from 04_feature_engineering.ipynb) ────────────

AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]
AGE_MAP = {age: i for i, age in enumerate(AGE_ORDER)}

LOW_RISK_AGES   = {"[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"}
MEDIUM_RISK_AGES = {"[50-60)", "[60-70)"}

MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

GLU_MAP = {'No': 0, 'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
A1C_MAP = {'No': 0, 'None': 0, 'Norm': 1, '>7': 2, '>8': 3}


def map_diag_category(diag):
    try:
        d = float(diag)
        if 390 <= d < 460:
            return 'circulatory'
        elif 460 <= d < 520:
            return 'respiratory'
        elif 520 <= d < 580:
            return 'digestive'
        elif 250 <= d < 251:
            return 'diabetes'
        else:
            return 'other'
    except Exception:
        return 'other'


def map_primary_diagnosis_group(code):
    try:
        d = float(code)
        if 390 <= d < 460:
            return 'Circulatory'
        elif 460 <= d < 520:
            return 'Respiratory'
        elif 520 <= d < 580:
            return 'Digestive'
        elif 250 <= d < 251:
            return 'Diabetes'
        elif 800 <= d < 1000:
            return 'Injury'
        else:
            return 'Other'
    except Exception:
        return 'Other'


def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Convert raw user inputs into the 39 model features.
    raw keys expected:
        age, diag_1, diag_2, diag_3,
        time_in_hospital, num_lab_procedures, num_procedures,
        num_medications, number_outpatient, number_emergency,
        number_inpatient, number_diagnoses,
        max_glu_serum, A1Cresult,
        insulin, change, diabetesMed,
        metformin, repaglinide, nateglinide, chlorpropamide,
        glimepiride, acetohexamide, glipizide, glyburide,
        tolbutamide, pioglitazone, rosiglitazone, acarbose,
        miglitol, troglitazone, tolazamide, examide, citoglipton,
        glyburide-metformin, glipizide-metformin,
        glimepiride-pioglitazone, metformin-rosiglitazone,
        metformin-pioglitazone
    """
    r = raw  # shorthand

    # ── Age features ─────────────────────────────────────────────────────────
    age = r.get('age', '[60-70)')
    age_ordinal = AGE_MAP.get(age, 5)
    if age in LOW_RISK_AGES:
        age_risk_group = 'low'
    elif age in MEDIUM_RISK_AGES:
        age_risk_group = 'medium'
    else:
        age_risk_group = 'high'

    # ── Diagnosis features ────────────────────────────────────────────────────
    diag_1 = str(r.get('diag_1', '0'))
    diag_2 = str(r.get('diag_2', '0'))
    diag_3 = str(r.get('diag_3', '0'))

    primary_group = map_primary_diagnosis_group(diag_1)
    # Rare groups collapse to "Other" (matching notebook behaviour)
    if primary_group in ('Digestive', 'Injury'):
        primary_group = 'Other'
    primary_diagnosis_group_reduced = primary_group

    d1g = map_diag_category(diag_1)
    d2g = map_diag_category(diag_2)
    d3g = map_diag_category(diag_3)

    has_circulatory  = int('circulatory'  in (d1g, d2g, d3g))
    has_respiratory  = int('respiratory'  in (d1g, d2g, d3g))
    has_diabetes_diag = int('diabetes'    in (d1g, d2g, d3g))

    all_groups = {d1g, d2g, d3g}
    num_unique_diag_groups = len(all_groups)
    num_non_other_diag = sum(g != 'other' for g in [d1g, d2g, d3g])

    # ── Numeric clinical features ─────────────────────────────────────────────
    time_in_hospital   = int(r.get('time_in_hospital', 3))
    num_lab_procedures = int(r.get('num_lab_procedures', 40))
    num_procedures     = int(r.get('num_procedures', 1))
    num_medications    = int(r.get('num_medications', 15))
    number_outpatient  = int(r.get('number_outpatient', 0))
    number_emergency   = int(r.get('number_emergency', 0))
    number_inpatient   = int(r.get('number_inpatient', 0))
    number_diagnoses   = int(r.get('number_diagnoses', 5))

    prior_inpatient_flag = int(number_inpatient >= 1)

    # ── Medication features ───────────────────────────────────────────────────
    insulin_val  = r.get('insulin', 'No')
    change_val   = r.get('change', 'No')
    diabmed_val  = r.get('diabetesMed', 'No')

    on_insulin        = int(insulin_val != 'No')
    med_change_flag   = int(change_val == 'Ch')
    diabetes_med_flag = int(diabmed_val == 'Yes')

    med_values = {col: r.get(col, 'No') for col in MED_COLS}

    num_active_medications = sum(v != 'No' for v in med_values.values())
    insulin_active         = int(med_values['insulin'] != 'No')
    med_change_intensity   = sum(v in ('Up', 'Down') for v in med_values.values())
    med_stable             = int(med_change_intensity == 0)

    medication_burden        = num_active_medications
    medication_burden_bucket = min(medication_burden, 3)

    # ── Utilisation features ──────────────────────────────────────────────────
    total_visits_raw = number_outpatient + number_emergency + number_inpatient
    total_visits     = max(total_visits_raw, 1)  # avoid division by zero

    emergency_ratio = number_emergency / total_visits
    inpatient_ratio = number_inpatient / total_visits
    visit_intensity = total_visits / max(time_in_hospital, 1)
    high_utilization = int(number_inpatient >= 2)

    # ── Clinical complexity ───────────────────────────────────────────────────
    procedure_density    = num_procedures / max(time_in_hospital, 1)
    diagnosis_complexity = number_diagnoses

    # ── Lab features ──────────────────────────────────────────────────────────
    glu_raw = str(r.get('max_glu_serum', 'No'))
    a1c_raw = str(r.get('A1Cresult', 'No'))

    high_glucose_flag = int(glu_raw in ('>200', '>300'))
    high_A1C_flag     = int(a1c_raw in ('>7', '>8'))
    glu_level         = GLU_MAP.get(glu_raw, 0)
    a1c_level         = A1C_MAP.get(a1c_raw, 0)

    # ── Interaction features ──────────────────────────────────────────────────
    meds_x_time      = num_medications * time_in_hospital
    inpatient_x_meds = number_inpatient * num_medications
    labs_x_time      = num_lab_procedures * time_in_hospital
    age_x_meds       = age_ordinal * num_medications
    inpatient_x_time = number_inpatient * time_in_hospital

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    row = {
        'primary_diagnosis_group_reduced': primary_diagnosis_group_reduced,
        'age_ordinal':              age_ordinal,
        'age_risk_group':           age_risk_group,
        'time_in_hospital':         time_in_hospital,
        'num_lab_procedures':       num_lab_procedures,
        'num_medications':          num_medications,
        'number_emergency':         number_emergency,
        'number_outpatient':        number_outpatient,
        'number_inpatient':         number_inpatient,
        'prior_inpatient_flag':     prior_inpatient_flag,
        'on_insulin':               on_insulin,
        'med_change_flag':          med_change_flag,
        'diabetes_med_flag':        diabetes_med_flag,
        'medication_burden_bucket': medication_burden_bucket,
        'total_visits':             total_visits,
        'emergency_ratio':          emergency_ratio,
        'inpatient_ratio':          inpatient_ratio,
        'visit_intensity':          visit_intensity,
        'high_utilization':         high_utilization,
        'procedure_density':        procedure_density,
        'diagnosis_complexity':     diagnosis_complexity,
        'high_glucose_flag':        high_glucose_flag,
        'high_A1C_flag':            high_A1C_flag,
        'glu_level':                glu_level,
        'a1c_level':                a1c_level,
        'has_circulatory':          has_circulatory,
        'has_respiratory':          has_respiratory,
        'has_diabetes_diag':        has_diabetes_diag,
        'num_unique_diag_groups':   num_unique_diag_groups,
        'num_non_other_diag':       num_non_other_diag,
        'num_active_medications':   num_active_medications,
        'insulin_active':           insulin_active,
        'med_change_intensity':     med_change_intensity,
        'med_stable':               med_stable,
        'meds_x_time':              meds_x_time,
        'inpatient_x_meds':         inpatient_x_meds,
        'labs_x_time':              labs_x_time,
        'age_x_meds':               age_x_meds,
        'inpatient_x_time':         inpatient_x_time,
    }

    return pd.DataFrame([row])[feature_columns]


# ── Prediction helpers ─────────────────────────────────────────────────────────

def _predict_one(name, model, X_raw_df, X_processed, threshold):
    """
    CatBoost is called with the raw DataFrame (it handles its own categoricals).
    RF and Stacking are called with the pre-processed numpy array.
    """
    if name == "catboost":
        prob = float(model.predict_proba(X_raw_df)[0][1])
    else:
        prob = float(model.predict_proba(X_processed)[0][1])
    label = int(prob >= threshold)
    return name, prob, label


def run_parallel_predictions(X_raw_df, X_processed):
    tasks = [
        ("catboost",      catboost_model, float(thresholds["catboost"])),
        ("random_forest", rf_model,       float(thresholds["random_forest"])),
        ("stacking",      stacking_model, float(thresholds["stacking"])),
    ]
    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(_predict_one, name, model, X_raw_df, X_processed, thr): name
            for name, model, thr in tasks
        }
        for future in as_completed(futures):
            name, prob, label = future.result()
            results[name] = {"probability": round(prob * 100, 2), "label": label}
    return results


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw = request.get_json(force=True)

        # 1. Engineer 39 features from raw inputs
        X_raw = engineer_features(raw)

        # 2. Cast categoricals so sklearn pipeline is happy
        for col in CAT_FEATURE_COLS:
            X_raw[col] = X_raw[col].astype(object)

        # 3. Pre-process for RF and Stacking
        X_processed = pipeline.transform(X_raw)

        # 4. Run three models in parallel
        results = run_parallel_predictions(X_raw, X_processed)

        return jsonify({"status": "ok", "predictions": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
