import pandas as pd

def calculate_risk(df):
    """Compute overall cyber risk score from key indicators"""
    df["Impact"] = df.iloc[:, 0].abs() if "Impact" not in df else df["Impact"]
    df["Exploitability"] = df.iloc[:, 1].abs() if "Exploitability" not in df else df["Exploitability"]

    df["Likelihood"] = (df["Impact"] + df["Exploitability"]) / 2
    df["Risk_Score"] = df["Likelihood"] * df["Impact"]

    def classify(score):
        if score < 0.3: return "Low"
        elif score < 0.6: return "Medium"
        else: return "High"

    df["Risk_Level"] = df["Risk_Score"].apply(classify)
    print("[INFO] Risk ranking calculated.")
    return df
