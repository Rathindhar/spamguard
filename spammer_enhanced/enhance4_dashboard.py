# ============================================================
#  ENHANCEMENT 4 — Flask Web Dashboard (Fixed)
#
#  Fixes applied:
#  1. debug=True caused Flask reloader to restart process,
#     skipping load_model() → model_data stayed None.
#     Fixed by using use_reloader=False.
#  2. load_model() now prints the exact path it's checking
#     so failures are always visible in the terminal.
#  3. Model loading errors are caught and printed clearly.
# ============================================================

import os
import re
import sys
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# ── Ensure imports work regardless of working directory ──────
BASE_DIR_APP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR_APP)
os.chdir(BASE_DIR_APP)   # set CWD immediately at import time

COMBINED_PATH = os.path.join(BASE_DIR_APP, "data", "combined_model.pkl")

# Must import so pickle can deserialise saved model objects
from enhance2_combined_model import StackedEnsemble, ConceptDriftDetector  # noqa

app        = Flask(__name__, template_folder=os.path.join(BASE_DIR_APP, "templates"))
model_data = None


# ════════════════════════════════════════════════════════════
#  Text cleaning helper
# ════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════
_fresh_scaler     = None
_fresh_feat_names = None


def _build_fresh_scaler():
    global _fresh_scaler, _fresh_feat_names
    from enhance1_behavioral_features import extract_behavioral_features
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    sample            = extract_behavioral_features("test message")
    _fresh_feat_names = list(sample.keys())

    clean_csv = os.path.join(BASE_DIR_APP, "data", "cleaned_data.csv")
    if os.path.exists(clean_csv):
        df   = pd.read_csv(clean_csv)
        rows = df["clean_message"].fillna("").apply(extract_behavioral_features)
        fd   = pd.DataFrame(list(rows))[_fresh_feat_names]
        fd   = fd.replace([float("inf"), float("-inf")], 0).fillna(0)
        _fresh_scaler = MinMaxScaler()
        _fresh_scaler.fit(fd)
    else:
        _fresh_scaler = MinMaxScaler()
        _fresh_scaler.fit(np.zeros((2, len(_fresh_feat_names))))

    print(f"   Scaler ready ({len(_fresh_feat_names)} features)")


def load_model():
    global model_data
    print(f"\n🔍 Looking for model at: {COMBINED_PATH}")
    print(f"   File exists: {os.path.exists(COMBINED_PATH)}")

    if not os.path.exists(COMBINED_PATH):
        print("⚠️  Model not found — run enhance1 → enhance2 first.")
        model_data = None
        return

    try:
        with open(COMBINED_PATH, "rb") as f:
            model_data = pickle.load(f)
        print("✅ Model loaded successfully")
        print(f"   Results: {model_data['results']}")
        _build_fresh_scaler()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model_data = None


# ════════════════════════════════════════════════════════════
#  Prediction
# ════════════════════════════════════════════════════════════
def predict_message(message: str) -> dict:
    if model_data is None:
        return {"error": "Model not loaded. Run training scripts first."}

    from enhance1_behavioral_features import extract_behavioral_features

    ensemble   = model_data["ensemble"]
    vectorizer = model_data["vectorizer"]
    drift_det  = model_data["drift_detector"]

    cleaned  = clean_text(message)
    X_tfidf  = vectorizer.transform([cleaned])
    bfeats   = extract_behavioral_features(message)
    X_b_raw  = np.array([[bfeats[f] for f in _fresh_feat_names]], dtype=float)
    X_b_raw  = np.nan_to_num(X_b_raw, nan=0.0, posinf=0.0, neginf=0.0)

    if _fresh_scaler.n_features_in_ != len(_fresh_feat_names):
        from sklearn.preprocessing import MinMaxScaler as _MMS
        tmp = _MMS(); tmp.fit(X_b_raw); X_b = tmp.transform(X_b_raw)
    else:
        X_b = _fresh_scaler.transform(X_b_raw)

    proba      = ensemble.predict_proba(X_tfidf, X_b)[0]
    label_num = int(proba[1] >= 0.35)
    spam_pct   = round(float(proba[1]) * 100, 1)
    ham_pct    = round(float(proba[0]) * 100, 1)

    feat_out  = vectorizer.get_feature_names_out()
    tfidf_arr = X_tfidf.toarray()[0]
    top_words = [(feat_out[i], round(float(tfidf_arr[i]), 4))
                 for i in np.argsort(tfidf_arr)[::-1]
                 if tfidf_arr[i] > 0][:8]

    top_behav = sorted(
        [(k, round(float(v), 3)) for k, v in bfeats.items() if v > 0],
        key=lambda x: -x[1]
    )[:8]

    drift_det.update(True)

    return {
        "message"      : message,
        "prediction"   : "SPAM" if label_num == 1 else "LEGITIMATE",
        "label_num"    : label_num,
        "spam_prob"    : spam_pct,
        "ham_prob"     : ham_pct,
        "top_words"    : top_words,
        "top_behaviors": top_behav,
        "drift_status" : drift_det.status(),
    }


# ════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════
@app.route("/")
def index():
    metrics = model_data["results"] if model_data else {}
    drift   = model_data["drift_detector"].status() if model_data else "Model not loaded"
    return render_template("index.html", metrics=metrics, drift=drift)


@app.route("/predict", methods=["POST"])
def predict():
    data    = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    return jsonify(predict_message(message))


@app.route("/batch", methods=["POST"])
def batch_predict():
    data     = request.get_json()
    messages = data.get("messages", [])
    results  = [predict_message(m) for m in messages if m.strip()]
    return jsonify({
        "total"     : len(results),
        "spam"      : sum(1 for r in results if r.get("label_num") == 1),
        "legitimate": sum(1 for r in results if r.get("label_num") == 0),
        "results"   : results,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data    = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Provide 'message' field"}), 400
    result = predict_message(message)
    return jsonify({
        "message"   : result["message"],
        "prediction": result["prediction"],
        "spam_prob" : result["spam_prob"],
        "ham_prob"  : result["ham_prob"],
    })


@app.route("/metrics")
def metrics():
    if model_data is None:
        return jsonify({"error": "Model not loaded"})
    return jsonify({
        "model_performance": model_data["results"],
        "individual_scores": model_data.get("individual_scores", {}),
        "drift_status"     : model_data["drift_detector"].status(),
    })


# ════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_model()
    print("\n" + "=" * 50)
    print("  🚀 Spammer Detection Dashboard")
    print("  Open your browser: http://localhost:5000")
    print("=" * 50 + "\n")
    # ── FIX: use_reloader=False prevents Flask from restarting
    #    the process, which would skip load_model() entirely
    app.run(debug=False, port=5000, use_reloader=False)