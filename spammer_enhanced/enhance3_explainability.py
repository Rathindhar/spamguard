# ============================================================
#  ENHANCEMENT 3 — Explainability (XAI)
#
#  "Why did the model flag this as spam?"
#  This module answers that question using two approaches:
#
#  A) Feature Importance — which behavioral features matter most
#     globally across all predictions (bar chart)
#
#  B) Per-message explanation — for any single message, shows
#     which words AND behavioral features triggered the verdict
# ============================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler

# Must import so pickle can deserialise saved model objects
from enhance2_combined_model import StackedEnsemble, ConceptDriftDetector  # noqa

COMBINED_PATH = os.path.join("data", "combined_model.pkl")
CLEAN_PATH    = os.path.join("data", "cleaned_data.csv")
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

STOP_WORDS = {
    "a","an","the","and","or","but","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will",
    "would","shall","should","may","might","must","can","could",
    "not","no","i","me","my","we","our","you","your","he","she",
    "it","its","they","them","their","this","that","to","of","in",
    "for","with","on","at","by","from","up","out","s","t","ll",
}

def clean_text(text):
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def build_fresh_scaler(feat_names):
    """
    Build a fresh MinMaxScaler from the cleaned dataset.
    This avoids any dimension mismatch from old saved pkl files.
    """
    from enhance1_behavioral_features import extract_behavioral_features
    df      = pd.read_csv(CLEAN_PATH)
    rows    = df["clean_message"].fillna("").apply(extract_behavioral_features)
    feat_df = pd.DataFrame(list(rows))[feat_names]
    feat_df = feat_df.replace([float('inf'), float('-inf')], 0).fillna(0)
    scaler  = MinMaxScaler()
    scaler.fit(feat_df)
    return scaler


def explain_prediction(message: str, model_data: dict, fresh_scaler) -> dict:
    """
    For a single message, return prediction + explanation.
    Uses fresh_scaler to avoid dimension mismatch with old pkl files.
    """
    from enhance1_behavioral_features import extract_behavioral_features

    ensemble   = model_data["ensemble"]
    vectorizer = model_data["vectorizer"]

    # Always get feat_names from the extractor itself (never from pkl)
    sample     = extract_behavioral_features("test message")
    feat_names = list(sample.keys())

    # ── Prepare features ─────────────────────────────────────
    cleaned  = clean_text(message)
    X_tfidf  = vectorizer.transform([cleaned])
    bfeats   = extract_behavioral_features(message)
    X_b_raw  = np.array([[bfeats[f] for f in feat_names]], dtype=float)
    X_b_raw  = np.nan_to_num(X_b_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_b      = fresh_scaler.transform(X_b_raw)

    # ── Predict ──────────────────────────────────────────────
    proba     = ensemble.predict_proba(X_tfidf, X_b)[0]
    label_num = int(proba[1] >= 0.5)
    spam_pct  = round(float(proba[1]) * 100, 1)
    ham_pct   = round(float(proba[0]) * 100, 1)

    # ── Top TF-IDF words ──────────────────────────────────────
    feat_out  = vectorizer.get_feature_names_out()
    tfidf_arr = X_tfidf.toarray()[0]
    top_words = [(feat_out[i], round(float(tfidf_arr[i]), 4))
                 for i in np.argsort(tfidf_arr)[::-1]
                 if tfidf_arr[i] > 0][:10]

    # ── Top behavioral features ───────────────────────────────
    top_behav = sorted(
        [(k, round(float(v), 3)) for k, v in bfeats.items() if v > 0],
        key=lambda x: -x[1]
    )[:8]

    return {
        "message"      : message,
        "prediction"   : "SPAM" if label_num == 1 else "LEGITIMATE",
        "confidence"   : spam_pct if label_num == 1 else ham_pct,
        "spam_prob"    : spam_pct,
        "ham_prob"     : ham_pct,
        "top_words"    : top_words,
        "top_behaviors": top_behav,
        "raw_bfeats"   : bfeats,
    }


def plot_global_feature_importance(model_data):
    print("\n📊 Generating global feature importance chart ...")
    from enhance1_behavioral_features import extract_behavioral_features

    df = pd.read_csv(CLEAN_PATH)
    df["label_num"] = df["label"].map({"spam": 1, "ham": 0})

    rows       = df["clean_message"].fillna("").apply(extract_behavioral_features)
    feat_df    = pd.DataFrame(list(rows))

    # Use only real feature names from extractor
    sample     = extract_behavioral_features("test")
    feat_names = list(sample.keys())
    feat_df    = feat_df[feat_names].replace([float('inf'),float('-inf')],0).fillna(0)

    spam_means = feat_df[df["label_num"] == 1].mean()
    ham_means  = feat_df[df["label_num"] == 0].mean()

    max_vals = feat_df.max().replace(0, 1)
    spam_norm = spam_means / max_vals
    ham_norm  = ham_means  / max_vals
    diff      = (spam_norm - ham_norm).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["#E53935" if v > 0 else "#43A047" for v in diff.values]
    ax.barh(diff.index, diff.values, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Behavioral Feature Importance\n"
        "(Red = Spam indicator | Green = Legitimate indicator)",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_xlabel("Normalised difference (Spam mean - Ham mean)", fontsize=10)
    red_patch   = mpatches.Patch(color="#E53935", alpha=0.85, label="Higher in Spam")
    green_patch = mpatches.Patch(color="#43A047", alpha=0.85, label="Higher in Legitimate")
    ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(OUTPUT_FOLDER, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   ✅ Saved: {path}")


def plot_message_explanation(result: dict, save_name="explanation.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    words  = [w[0] for w in result["top_words"][:8]]
    scores = [w[1] for w in result["top_words"][:8]]
    color  = "#E53935" if result["prediction"] == "SPAM" else "#43A047"

    ax1.barh(words[::-1], scores[::-1], color=color, alpha=0.8)
    ax1.set_title("Top Content Words (TF-IDF weight)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("TF-IDF Score")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.spines[["top","right"]].set_visible(False)

    bnames  = [b[0].replace("_", " ") for b in result["top_behaviors"][:8]]
    bscores = [b[1] for b in result["top_behaviors"][:8]]
    bcolors = ["#E53935" if result["prediction"]=="SPAM" else "#43A047"] * len(bnames)

    ax2.barh(bnames[::-1], bscores[::-1], color=bcolors[::-1], alpha=0.8)
    ax2.set_title("Top Behavioral Features (raw values)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Feature Value")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.spines[["top","right"]].set_visible(False)

    verdict = result["prediction"]
    conf    = result["confidence"]
    emoji   = "SPAM" if verdict == "SPAM" else "HAM"
    fig.suptitle(
        f'[{emoji}]  Confidence: {conf:.1f}%\n'
        f'"{result["message"][:70]}{"..." if len(result["message"])>70 else ""}"',
        fontsize=11, fontweight="bold",
        color="#C62828" if verdict=="SPAM" else "#2E7D32"
    )

    path = os.path.join(OUTPUT_FOLDER, save_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_explainability():
    print("=" * 58)
    print("  ENHANCEMENT 3 — Explainability (XAI)")
    print("=" * 58)

    if not os.path.exists(COMBINED_PATH):
        print(f"\n❌ Run enhance2_combined_model.py first.")
        return

    with open(COMBINED_PATH, "rb") as f:
        model_data = pickle.load(f)

    # Build fresh scaler — always correct dimensions, no pkl mismatch
    from enhance1_behavioral_features import extract_behavioral_features
    sample     = extract_behavioral_features("test")
    feat_names = list(sample.keys())
    print(f"\n🔧 Building fresh scaler for {len(feat_names)} features ...")
    fresh_scaler = build_fresh_scaler(feat_names)
    print(f"   ✅ Scaler ready (fitted on {len(feat_names)} features)")

    # Global feature importance chart
    plot_global_feature_importance(model_data)

    # Per-message explanations
    test_messages = [
        "Congratulations! You've won a FREE iPhone! Click here to claim your $1000 prize NOW!!!",
        "Hey, are we still on for lunch tomorrow at the usual place?",
        "URGENT: Your bank account has been SUSPENDED. Call 0800-FREE to restore access immediately!",
    ]

    print("\n🔍 Explaining individual predictions:\n")
    for i, msg in enumerate(test_messages):
        result = explain_prediction(msg, model_data, fresh_scaler)
        print(f"  Message  : \"{msg[:65]}...\"")
        print(f"  Verdict  : {result['prediction']}  "
              f"(Spam prob: {result['spam_prob']:.1f}%)")
        print(f"  Top words: {', '.join([w[0] for w in result['top_words'][:5]])}")
        rb = result['raw_bfeats']
        print(f"  Behaviors: spam_keywords={rb['spam_keyword_score']:.0f}, "
              f"caps_ratio={rb['uppercase_ratio']:.2f}, "
              f"exclamations={rb['exclamation_count']:.0f}")
        plot_message_explanation(result, f"explanation_msg{i+1}.png")
        print(f"  📊 Chart → results/explanation_msg{i+1}.png\n")

    print("✅ Explainability analysis complete!")
    print("\nNext → run:  python enhance4_dashboard.py")


if __name__ == "__main__":
    run_explainability()
