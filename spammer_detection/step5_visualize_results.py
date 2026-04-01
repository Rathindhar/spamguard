# ============================================================
#  STEP 5 — Visualise Results
#  Generates charts matching the paper's results section:
#  • Bar chart comparing all 4 classifiers (Table I style)
#  • Confusion matrix for the best model (Table II style)
# ============================================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── File paths ───────────────────────────────────────────────
MODELS_PATH   = os.path.join("data", "trained_models.pkl")
FEATURES_PATH = os.path.join("data", "features.pkl")
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def visualize():
    print("=" * 55)
    print("  Spammer Identification — Visualising Results")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────
    for path, name in [(MODELS_PATH, "Trained models"), (FEATURES_PATH, "Features")]:
        if not os.path.exists(path):
            print(f"\n❌ {name} not found at '{path}'")
            print("   Make sure you ran all previous steps first.")
            return

    with open(MODELS_PATH,   "rb") as f: model_data   = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feature_data = pickle.load(f)

    results   = model_data["results"]
    models    = model_data["models"]
    best_name = model_data["best"]
    X_test    = feature_data["X_test"]
    y_test    = feature_data["y_test"]

    # ── CHART 1: Grouped bar chart (all metrics, all models) ─
    print("\n📊 Creating performance comparison chart ...")

    classifiers = list(results.keys())
    metrics     = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors      = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    x     = np.arange(len(classifiers))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[clf][metric] for clf in classifiers]
        bars = ax.bar(x + i * width, values, width, label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_title("Performance Comparison of Machine Learning Models\n(Spammer Identification in Industrial Mobile Cloud Systems)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Classifier", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(classifiers, fontsize=9)
    ax.set_ylim(85, 102)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    chart1_path = os.path.join(OUTPUT_FOLDER, "performance_comparison.png")
    plt.tight_layout()
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"   ✅ Saved to: {chart1_path}")

    # ── CHART 2: Confusion Matrix for best model ──────────────
    print(f"\n📊 Creating confusion matrix for: {best_name} ...")

    best_model = models[best_name]
    y_pred     = best_model.predict(X_test)
    cm         = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate (Ham)", "Spammer (Spam)"]
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}\n(Best Performing Model)",
                 fontsize=12, fontweight="bold", pad=12)

    chart2_path = os.path.join(OUTPUT_FOLDER, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print(f"   ✅ Saved to: {chart2_path}")

    # ── Print final summary table ─────────────────────────────
    print("\n" + "=" * 57)
    print("  TABLE I — Performance Comparison (matches paper)")
    print("=" * 57)
    print(f"  {'Classifier':<24} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("─" * 57)
    for clf in classifiers:
        r = results[clf]
        marker = " ◀ best" if clf == best_name else ""
        print(f"  {clf:<24} {r['Accuracy']:>5.1f}% {r['Precision']:>5.1f}%"
              f" {r['Recall']:>5.1f}% {r['F1-Score']:>5.1f}%{marker}")
    print("=" * 57)

    print(f"\n✅ All charts saved in '{OUTPUT_FOLDER}/' folder")
    print("\nNext step → run:  python step6_predict.py")


if __name__ == "__main__":
    visualize()
