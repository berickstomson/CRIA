import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_risk_distribution(df):
    plt.figure(figsize=(6,4))
    df["Risk_Level"].value_counts().plot(kind="bar", color="steelblue")
    plt.title("Cyber Risk Level Distribution")
    plt.tight_layout()
    plt.show()
    plt.savefig("../results/risk_distribution.png")
    print("[INFO] Risk distribution chart saved.")
    plt.close()

def plot_model_comparison(metrics_path="../results/metrics_report.json"):
    with open(metrics_path) as f:
        metrics = json.load(f)

    models = list(metrics.keys())
    accuracy_scores = [metrics[m]["accuracy"] for m in models]

    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracy_scores, color=["#7fc97f", "#beaed4"])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)  # Set y-axis limit to be slightly above 1.0
    plt.show()
    plt.savefig("../results/accuracy_comparison.png")
    print("[INFO] Accuracy comparison chart saved.")
    plt.close()
