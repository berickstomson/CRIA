print("--- RUNNING NEW VERSION OF centralized_training.py ---") # <-- New check
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib, json, os

def train_models(X, y):
    """Train centralized ML models for risk classification"""
    # --- Start of robust path handling ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_dir = os.path.join(project_root, "models")
    # Ensure the save directories exist
    # --- End of robust path handling ---
    os.makedirs(save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "xgboost_model.pkl": XGBClassifier(eval_metric="logloss"),
        "lightgbm_model.pkl": LGBMClassifier(verbosity=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, preds)
        }
        results[name] = metrics
        joblib.dump(model, os.path.join(save_dir, name))
        print(f"[INFO] {name} trained and saved.")

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Debugging: Print the results dictionary before saving
    print("[DEBUG] Final results dictionary:")
    print(json.dumps(results, indent=4))

    # --- Start of robust file writing ---
    file_path = os.path.join(results_dir, "metrics_report.json")
    print(f"[DEBUG] Attempting to write to: {os.path.abspath(file_path)}")
    try:
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
            f.flush()
        print("[DEBUG] File write operation completed.")
    except Exception as e:
        print(f"[ERROR] Failed to write to file: {e}")
    # --- End of robust file writing ---

    print("[INFO] Centralized training complete.")
