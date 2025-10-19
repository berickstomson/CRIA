import numpy as np
from xgboost import XGBClassifier
from sklearn import svm
import pandas as pd
import joblib, os

def federated_training(X, y, save_dir="../models"):
    """Simulated Federated Learning for IIoT risk prediction"""
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Combine X and y to split them together, maintaining alignment
    df = pd.concat([X, y], axis=1)
    target_col = y.name

    splits = np.array_split(df, 3) # Simulate 3 clients
    clients = []

    for i, local_df in enumerate(splits):
        X_local = local_df.drop(columns=[target_col])
        y_local = local_df[target_col]
        model = XGBClassifier(eval_metric="logloss") if i % 2 == 0 else svm.SVC()
        model.fit(X_local, y_local)
        clients.append(model)
        print(f"[INFO] Client {i+1} local training done.")

    joblib.dump(clients[0], os.path.join(save_dir, "fedxgboost_model.pkl"))
    joblib.dump(clients[1], os.path.join(save_dir, "fedsvm_model.pkl"))
    print("[INFO] Federated training simulation completed.")
