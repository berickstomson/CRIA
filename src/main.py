from data_preprocessing import load_all_datasets, preprocess_data
from threat_modeling_stride import generate_threat_profile
from centralized_training import train_models
from federated_training import federated_training
from risk_ranking import calculate_risk
from visualization import plot_risk_distribution, plot_model_comparison
from security_controls import suggest_controls
import pandas as pd

if __name__ == "__main__":
    print("=== Starting CIRA-ML Risk Assessment ===")

    df = load_all_datasets("data")
    df = preprocess_data(df)
    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing. All rows may have been dropped due to missing values.")

    df, summary = generate_threat_profile(df)

    # --- Centralized Data Preparation ---
    target_col = None
    for col in df.columns:
        if "risk" in col.lower() or "label" in col.lower():
            target_col = col
            break
    if not target_col:
        df["Risk_Label"] = (df.iloc[:, 0].rank(pct=True) > 0.5).astype(int)
        target_col = "Risk_Label"

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)

    train_models(X, y)
    federated_training(X, y)

    risk_df = calculate_risk(df) # Assuming this function needs the original df with target
    risk_df.to_csv("../results/risk_table.csv", index=False)

    plot_risk_distribution(risk_df)
    plot_model_comparison()

    print(f"High Risk Controls: {suggest_controls('High')}")
    print("=== Completed Successfully ===")
