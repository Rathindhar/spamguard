# ============================================================
#  STEP 6 — Predict on New Messages (Live Demo)
#  This is the "Spammer Identification Process" from
#  Section IV-E of the paper.
#
#  You can type any message and the best trained model
#  will tell you if it looks like spam or legitimate.
# ============================================================

import os
import re
import pickle

# ── File paths ───────────────────────────────────────────────
MODELS_PATH   = os.path.join("data", "trained_models.pkl")
FEATURES_PATH = os.path.join("data", "features.pkl")

# ── Same stop-words list used in preprocessing ───────────────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "shall", "should", "may",
    "might", "must", "can", "could", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very",
    "just", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "through",
    "during", "before", "after", "above", "below", "from",
    "up", "down", "in", "out", "on", "off", "then", "once",
    "here", "there", "when", "where", "why", "how", "all",
    "any", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "it", "its", "they", "them", "their", "this",
    "that", "these", "those", "to", "into", "what", "which",
    "who", "whom", "s", "t", "ll", "re", "ve", "d", "m",
}


def clean_text(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def predict(message: str, model, vectorizer) -> dict:
    """
    Takes a raw message string.
    Returns a dict with the prediction and confidence.
    """
    cleaned   = clean_text(message)
    vec       = vectorizer.transform([cleaned])
    label_num = model.predict(vec)[0]

    # Some models (LinearSVC) don't output probabilities;
    # we use decision_function as a confidence proxy instead.
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        confidence = prob[label_num] * 100
    elif hasattr(model, "decision_function"):
        score      = model.decision_function(vec)[0]
        # Convert to a 0–100 scale (rough estimate)
        confidence = min(100.0, max(0.0, 50 + score * 10))
    else:
        confidence = None

    label = "🚨 SPAMMER (Spam)" if label_num == 1 else "✅ LEGITIMATE (Ham)"
    return {"label": label, "label_num": label_num, "confidence": confidence}


def run_demo():
    print("=" * 55)
    print("  Spammer Identification — Live Prediction Demo")
    print("=" * 55)

    # ── Load models ───────────────────────────────────────────
    for path, name in [(MODELS_PATH, "Trained models"), (FEATURES_PATH, "Features")]:
        if not os.path.exists(path):
            print(f"\n❌ {name} not found at '{path}'")
            print("   Please run all previous steps first.")
            return

    with open(MODELS_PATH,   "rb") as f: model_data   = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feature_data = pickle.load(f)

    best_name  = model_data["best"]
    best_model = model_data["models"][best_name]
    vectorizer = feature_data["vectorizer"]

    print(f"\n✅ Using best model: {best_name}")
    print("\n" + "─" * 55)

    # ── Example predictions (like in the paper) ───────────────
    test_messages = [
        "Congratulations! You have won a FREE iPhone. Click here to claim your prize now!",
        "Hey, are we still meeting for lunch at 1pm today?",
        "URGENT: Your bank account has been suspended. Call us immediately to restore access.",
        "The project report is due on Friday. Let me know if you need help.",
        "Win $1000 cash prize! Text WIN to 80082 now. Limited time offer!!!",
        "Don't forget to bring your laptop to the team meeting tomorrow morning.",
    ]

    print("  Running predictions on example messages:\n")
    for msg in test_messages:
        result = predict(msg, best_model, vectorizer)
        print(f"  Message   : \"{msg[:60]}{'...' if len(msg)>60 else ''}\"")
        print(f"  Prediction: {result['label']}")
        if result["confidence"] is not None:
            print(f"  Confidence: {result['confidence']:.1f}%")
        print()

    # ── Interactive mode ──────────────────────────────────────
    print("─" * 55)
    print("  Interactive Mode — type your own messages!")
    print("  (Type 'quit' or 'exit' to stop)\n")

    while True:
        try:
            user_input = input("  Enter a message: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q", ""):
            print("\n  Goodbye! 👋")
            break

        result = predict(user_input, best_model, vectorizer)
        print(f"\n  ➤ Prediction : {result['label']}")
        if result["confidence"] is not None:
            print(f"  ➤ Confidence : {result['confidence']:.1f}%")
        print()


if __name__ == "__main__":
    run_demo()
