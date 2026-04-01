# ============================================================
#  STEP 1 — Download the Dataset
#  We use the SMS Spam Collection dataset (public, free)
#  It contains 5,574 SMS messages labeled as "spam" or "ham"
#  (ham = legitimate/normal message)
# ============================================================

import urllib.request
import os
import zipfile

# ── Where to save things ────────────────────────────────────
SAVE_FOLDER = "data"
os.makedirs(SAVE_FOLDER, exist_ok=True)

ZIP_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = os.path.join(SAVE_FOLDER, "smsspamcollection.zip")
RAW_PATH = os.path.join(SAVE_FOLDER, "SMSSpamCollection")

def download_dataset():
    print("=" * 55)
    print("  Spammer Identification — Dataset Download")
    print("=" * 55)

    # ── Check if already downloaded ─────────────────────────
    if os.path.exists(RAW_PATH):
        print(f"\n✅ Dataset already exists at: {RAW_PATH}")
        print("   No download needed. Move on to step2_preprocess.py")
        return

    # ── Download ─────────────────────────────────────────────
    print(f"\n📥 Downloading SMS Spam Collection ...")
    try:
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print(f"   Saved zip to: {ZIP_PATH}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Manual fix:")
        print("   1. Visit https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        print("   2. Download the zip file")
        print(f"   3. Extract 'SMSSpamCollection' into the '{SAVE_FOLDER}/' folder")
        return

    # ── Extract ──────────────────────────────────────────────
    print("📦 Extracting zip ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(SAVE_FOLDER)

    print(f"\n✅ Done!  Dataset ready at: {RAW_PATH}")
    print("\nNext step → run:  python step2_preprocess.py")


if __name__ == "__main__":
    download_dataset()
