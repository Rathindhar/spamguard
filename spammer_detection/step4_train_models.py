# ============================================================
#  STEP 4 — Train All 4 Machine Learning Models
#  As described in the paper (Section IV-D):
#  1. Naive Bayes (NB)
#  2. Logistic Regression (LR)
#  3. Random Forest (RF)
#  4. Support Vector Machine (SVM)
# ============================================================

import os
import pickle

from sklearn.naive_bayes       import MultinomialNB
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.svm               import LinearSVC
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score
)

# ── File paths ───────────────────────────────────────────────
FEATURES_PATH = os.path.join("data", "features.pkl")
MODELS_PATH   = os.path.join("data", "trained_models.pkl")


def train_and_evaluate():
    print("=" * 55)
    print("  Spammer Identification — Model Training & Evaluation")
    print("=" * 55)

    # ── Load feature data ────────────────────────────────────
    if not os.path.exists(FEATURES_PATH):
        print(f"\n❌ Features not found at '{FEATURES_PATH}'")
        print("   Please run step3_feature_extraction.py first.")
        return

    with open(FEATURES_PATH, "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    print(f"\n✅ Loaded feature data")
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Testing  samples : {X_test.shape[0]}")

    # ── Define all 4 classifiers ─────────────────────────────
    classifiers = {
        "Naive Bayes"         : MultinomialNB(),
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": LinearSVC(max_iter=2000, random_state=42),
    }

    results  = {}
    trained  = {}

    print("\n" + "─" * 55)
    print(f"  {'Classifier':<24} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("─" * 55)

    for name, clf in classifiers.items():
        # ── Train ──────────────────────────────────────────
        clf.fit(X_train, y_train)

        # ── Predict on unseen test data ───────────────────
        y_pred = clf.predict(X_test)

        # ── Calculate metrics ─────────────────────────────
        acc  = accuracy_score (y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec  = recall_score   (y_test, y_pred, zero_division=0) * 100
        f1   = f1_score       (y_test, y_pred, zero_division=0) * 100

        results[name] = {"Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1-Score": f1}
        trained[name] = clf

        print(f"  {name:<24} {acc:>5.1f}% {prec:>5.1f}% {rec:>5.1f}% {f1:>5.1f}%")

    print("─" * 55)

    # ── Identify the best model ───────────────────────────────
    best_name = max(results, key=lambda n: results[n]["F1-Score"])
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Accuracy  : {results[best_name]['Accuracy']:.1f}%")
    print(f"   Precision : {results[best_name]['Precision']:.1f}%")
    print(f"   Recall    : {results[best_name]['Recall']:.1f}%")
    print(f"   F1-Score  : {results[best_name]['F1-Score']:.1f}%")

    # ── Save trained models & results ─────────────────────────
    save_data = {
        "models" : trained,
        "results": results,
        "best"   : best_name,
    }
    with open(MODELS_PATH, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\n✅ Trained models saved to: {MODELS_PATH}")
    print("\nNext step → run:  python step5_visualize_results.py")


if __name__ == "__main__":
    train_and_evaluate()
