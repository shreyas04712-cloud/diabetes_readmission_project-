├── notebooks/
│   ├── 01_data_audit_and_target_definition.ipynb
│   ├── 02_eda_phase1_target_and_structure.ipynb
│   ├── 03_eda_phase2_clinical_and_age_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_building.ipynb
│   └── 06_model_evaluation_final.ipynb
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── app.py                 (Flask Web Application)
├── rebuild_artifacts.py   (Preprocessing/Artifact Utility)
├── retrain_models.py      (Local Model Retraining Script)
├── requirements.txt       (Project Dependencies)

**The Datasets cannot be uploaded due to massive file size , below is the dataset overview**

| Dataset                                 | Type                  | Purpose                 | Description                                                                                                                                                                   |
| --------------------------------------- | --------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **diabetic_data.csv**                   | Raw Dataset           | Primary source data     | Original hospital encounter records for diabetic patients. Contains demographics, diagnoses, laboratory results, medications, hospital utilization, and readmission outcomes. |
| **IDS_mapping.csv**                     | Raw Reference Dataset | Data dictionary         | Maps coded identifiers and categorical values from the raw dataset into meaningful descriptions and categories.                                                               |
| **diabetic_data_base_table.csv**        | Processed Dataset     | Cleaned base table      | Initial cleaned version of the raw dataset after preprocessing, duplicate handling, and target preparation. Used as the foundation for EDA and feature engineering.           |
| **diabetic_data_base_table_mapped.csv** | Processed Dataset     | Human-readable dataset  | Base table with categorical codes replaced by descriptive labels using the mapping file, improving interpretability during analysis.                                          |
| **missing_value_audit.csv**             | Audit Dataset         | Data quality assessment | Records missing-value counts, percentages, and treatment decisions for each variable during preprocessing.                                                                    |
| **diabetes_model_base.csv**             | Model Dataset         | Final modeling dataset  | Fully engineered dataset containing selected features and the target variable (`readmit_30`) used for machine learning model training.                                        |


