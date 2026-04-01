# ============================================================
#  STEP 2 — Data Pre-processing
#  Exactly as described in the paper (Section IV-B):
#  • Lowercase conversion
#  • Remove special characters, numbers, punctuation
#  • Tokenisation
#  • Stop-word removal
#  • Remove extra spaces / symbols
# ============================================================

import os
import re
import pandas as pd

# ── File paths ───────────────────────────────────────────────
RAW_PATH   = os.path.join("data", "SMSSpamCollection")
CLEAN_PATH = os.path.join("data", "cleaned_data.csv")

# ── Common English stop-words (no extra library needed) ──────
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
    "any", "both", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "it", "its", "they", "them", "their", "this",
    "that", "these", "those", "to", "into", "what", "which",
    "who", "whom", "s", "t", "ll", "re", "ve", "d", "m",
}


# ── Cleaning function ─────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Apply all pre-processing steps from the paper to one message.
    Returns a cleaned string ready for TF-IDF feature extraction.
    """
    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove special characters, numbers, punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 3 – tokenise (split into words)
    tokens = text.split()

    # Step 4 – remove stop-words
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # Step 5 – remove very short tokens (noise), rejoin
    tokens = [word for word in tokens if len(word) > 1]

    return " ".join(tokens)


def preprocess():
    print("=" * 55)
    print("  Spammer Identification — Pre-processing")
    print("=" * 55)

    # ── Load raw dataset ─────────────────────────────────────
    if not os.path.exists(RAW_PATH):
        print(f"\n❌ Dataset not found at '{RAW_PATH}'")
        print("   Please run step1_get_dataset.py first.")
        return

    print(f"\n📂 Loading raw data from: {RAW_PATH}")
    df = pd.read_csv(
        RAW_PATH,
        sep="\t",          # tab-separated
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )
    print(f"   Total messages loaded : {len(df)}")
    print(f"   Spam messages         : {(df['label'] == 'spam').sum()}")
    print(f"   Legitimate (ham)      : {(df['label'] == 'ham').sum()}")

    # ── Convert labels to numbers  (spam=1, ham=0) ───────────
    df["label_num"] = df["label"].map({"spam": 1, "ham": 0})

    # ── Apply cleaning ────────────────────────────────────────
    print("\n🧹 Cleaning messages ...")
    df["clean_message"] = df["message"].apply(clean_text)

    # ── Quick preview ─────────────────────────────────────────
    print("\n── Sample before / after cleaning ──────────────────")
    for idx in [0, 1, 2]:
        print(f"\n  Original : {df['message'].iloc[idx][:80]}")
        print(f"  Cleaned  : {df['clean_message'].iloc[idx][:80]}")

    # ── Save cleaned data ─────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df[["label", "label_num", "clean_message"]].to_csv(CLEAN_PATH, index=False)
    print(f"\n✅ Cleaned data saved to: {CLEAN_PATH}")
    print("\nNext step → run:  python step3_feature_extraction.py")


if __name__ == "__main__":
    preprocess()
