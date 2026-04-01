# ============================================================
#  STEP 3 — Feature Extraction using TF-IDF
#  Exactly as described in the paper (Section IV-C):
#  TF-IDF converts each cleaned message into a numerical
#  vector that machine learning algorithms can understand.
#
#  TF  = Term Frequency   (how often a word appears in a msg)
#  IDF = Inverse Document Frequency (how rare it is overall)
#  High TF-IDF → word is important & distinctive for that msg
# ============================================================

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── File paths ───────────────────────────────────────────────
CLEAN_PATH    = os.path.join("data", "cleaned_data.csv")
FEATURES_PATH = os.path.join("data", "features.pkl")   # saved feature data


def extract_features():
    print("=" * 55)
    print("  Spammer Identification — Feature Extraction (TF-IDF)")
    print("=" * 55)

    # ── Load cleaned data ─────────────────────────────────────
    if not os.path.exists(CLEAN_PATH):
        print(f"\n❌ Cleaned data not found at '{CLEAN_PATH}'")
        print("   Please run step2_preprocess.py first.")
        return

    df = pd.read_csv(CLEAN_PATH)
    print(f"\n📂 Loaded {len(df)} cleaned messages")

    X_text = df["clean_message"].fillna("")   # the text
    y      = df["label_num"]                  # 1=spam, 0=ham

    # ── Split into Train (80%) and Test (20%) ─────────────────
    # Random state=42 means you get the same split every time
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.20,
        random_state=42,
        stratify=y        # keep spam/ham ratio balanced
    )
    print(f"\n📊 Train set : {len(X_train_text)} messages")
    print(f"   Test set  : {len(X_test_text)} messages")

    # ── Build TF-IDF matrix ───────────────────────────────────
    print("\n🔢 Applying TF-IDF vectorisation ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,   # keep the 5000 most informative words
        ngram_range=(1, 2),  # use single words AND two-word pairs
        sublinear_tf=True    # dampen very high term frequencies
    )

    # IMPORTANT: fit only on training data to prevent data leakage
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)

    print(f"   Feature matrix shape — Train: {X_train.shape}")
    print(f"   Feature matrix shape — Test : {X_test.shape}")
    print(f"   Each message is now a vector of {X_train.shape[1]} numbers")

    # ── Save everything for the next step ────────────────────
    save_data = {
        "X_train"   : X_train,
        "X_test"    : X_test,
        "y_train"   : y_train,
        "y_test"    : y_test,
        "vectorizer": vectorizer,
    }
    os.makedirs("data", exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\n✅ Features saved to: {FEATURES_PATH}")
    print("\nNext step → run:  python step4_train_models.py")


if __name__ == "__main__":
    extract_features()
