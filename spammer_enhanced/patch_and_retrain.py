# ============================================================
#  patch_and_retrain.py
#  Run this ONCE from your spammer_enhanced folder.
#  It patches enhance1 + enhance2, clears old model files,
#  and retrains everything from scratch.
#  Usage:  python patch_and_retrain.py
# ============================================================

import os, sys, subprocess, re

THIS = os.path.dirname(os.path.abspath(__file__))

def patch_enhance1():
    path = os.path.join(THIS, "enhance1_behavioral_features.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # ── 1. Replace entire SPAM_KEYWORDS block ───────────────
    new_spam = '''SPAM_KEYWORDS = {
    # Classic spam
    "free", "win", "winner", "won", "prize", "claim", "urgent",
    "congratulations", "selected", "offer", "cash", "reward",
    "click", "call", "text", "subscribe", "guaranteed", "limited",
    "exclusive", "bonus", "discount", "deal", "cheap", "credit",
    "loan", "debt", "investment", "earn", "income", "profit",
    "account", "suspended", "verify", "confirm", "password",
    "bank", "transfer", "nigeria", "inheritance", "million",
    # Action/urgency words
    "activate", "download", "register", "apply", "update", "upgrade",
    "recharge", "cashback", "membership", "subscription", "scheme",
    "opportunity", "returns", "daily", "instantly", "immediately",
    "deactivated", "blocked", "pending", "failed", "suspicious",
    # Indian context spam
    "paytm", "kyc", "sim", "upi", "gpay", "phonepe", "crypto",
    "government", "subsidy", "netflix", "amazon", "iphone",
    "parcel", "delivery", "lucky", "draw", "vacation", "dubai",
    "laptop", "mobile", "data", "storage",
    # Soft spam words
    "money", "profits", "join", "premium", "secret", "online",
    "method", "trick", "fast", "quickly", "hours", "suspension",
    "interest", "zero", "google", "activity", "branded", "clothes",
    "worth", "available", "student", "email", "full",
    "card", "details", "simple", "app", "monthly",
}'''

    src = re.sub(
        r'SPAM_KEYWORDS\s*=\s*\{[^}]+\}',
        new_spam, src, flags=re.DOTALL
    )

    # ── 2. Replace entire HAM_KEYWORDS block ────────────────
    new_ham = '''HAM_KEYWORDS = {
    "professor", "lecture", "slides", "assignment", "exam", "class",
    "notes", "study", "college", "coding", "debug", "java", "github",
    "repository", "recursion", "interview", "cricket", "practice",
    "library", "birthday", "dinner", "movie", "charger", "meeting",
    "office", "project", "files", "results", "announced",
    "milk", "home", "bring", "reach", "lunch", "buy", "asked", "mom",
}'''

    if "HAM_KEYWORDS" in src:
        src = re.sub(
            r'HAM_KEYWORDS\s*=\s*\{[^}]+\}',
            new_ham, src, flags=re.DOTALL
        )
    else:
        # Insert after SPAM_KEYWORDS block
        src = src.replace(
            "ACTION_STARTERS",
            new_ham + "\n\nACTION_STARTERS"
        )

    # ── 3. Fix fallback return to use named keys ─────────────
    old_fallback = re.search(
        r'return \{f"feat_\{i\}".*?\}', src, re.DOTALL
    )
    if old_fallback:
        src = src[:old_fallback.start()] + (
            'return {\n'
            '            "msg_length": 0, "word_count": 0, "avg_word_length": 0,\n'
            '            "uppercase_ratio": 0, "digit_ratio": 0, "special_char_ratio": 0,\n'
            '            "exclamation_count": 0, "url_count": 0, "spam_keyword_score": 0,\n'
            '            "punctuation_density": 0, "unique_word_ratio": 0, "sentence_count": 0,\n'
            '            "avg_sentence_length": 0, "caps_word_count": 0, "question_mark_count": 0,\n'
            '            "number_count": 0, "currency_count": 0, "repeated_char_score": 0,\n'
            '            "lexical_diversity": 0, "starts_with_action": 0, "ham_keyword_score": 0,\n'
            '        }'
        ) + src[old_fallback.end():]

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("  ✅ enhance1_behavioral_features.py patched")


def patch_enhance2():
    path = os.path.join(THIS, "enhance2_combined_model.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # Lower threshold: replace 0.5 with 0.35 in predict method only
    # Use a targeted replacement inside StackedEnsemble class
    if "SPAM_THRESHOLD" not in src:
        old = "    def predict(self, X_tfidf, X_behavioral):\n        meta_features = self._get_meta_features(X_tfidf, X_behavioral)\n        return self.meta_learner.predict(meta_features)"
        new = ("    SPAM_THRESHOLD = 0.35  # catches subtle spam (lowered from 0.5)\n\n"
               "    def predict(self, X_tfidf, X_behavioral):\n"
               "        proba = self.predict_proba(X_tfidf, X_behavioral)\n"
               "        return (proba[:, 1] >= self.SPAM_THRESHOLD).astype(int)")
        if old in src:
            src = src.replace(old, new)
            print("  ✅ enhance2_combined_model.py threshold patched (0.5 → 0.35)")
        else:
            # Already patched or different format — check
            if "0.35" in src:
                print("  ✅ enhance2_combined_model.py threshold already at 0.35")
            else:
                print("  ⚠️  enhance2 predict pattern not found — skipping threshold patch")
    else:
        # Update existing threshold value
        src = re.sub(r'SPAM_THRESHOLD\s*=\s*[\d.]+', 'SPAM_THRESHOLD = 0.35', src)
        print("  ✅ enhance2_combined_model.py threshold updated to 0.35")

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


def patch_dashboard():
    path = os.path.join(THIS, "enhance4_dashboard.py")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    src = re.sub(r'proba\[1\]\s*>=\s*0\.[5-9]', 'proba[1] >= 0.35', src)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("  ✅ enhance4_dashboard.py threshold updated")


def delete_stale():
    data_dir = os.path.join(THIS, "data")
    deleted = []
    for fname in ["combined_model.pkl", "enhanced_features.pkl"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            deleted.append(fname)
    if deleted:
        print(f"  ✅ Deleted stale files: {', '.join(deleted)}")
    else:
        print("  ✅ No stale pkl files found")


def retrain():
    print("\n  Retraining model with patched files...")
    for script in ["enhance1_behavioral_features.py",
                   "enhance2_combined_model.py",
                   "enhance3_explainability.py"]:
        sp = os.path.join(THIS, script)
        if not os.path.exists(sp):
            print(f"  ⚠️  {script} not found, skipping")
            continue
        r = subprocess.run([sys.executable, sp], cwd=THIS)
        if r.returncode != 0:
            print(f"  ❌ {script} failed — check error above")
            sys.exit(1)
        print(f"  ✅ {script} done")


if __name__ == "__main__":
    print("=" * 55)
    print("  SpamGuard — Patch & Retrain")
    print("=" * 55)

    print("\n[1/4] Patching enhance1_behavioral_features.py ...")
    patch_enhance1()

    print("\n[2/4] Patching enhance2_combined_model.py ...")
    patch_enhance2()

    print("\n[3/4] Patching enhance4_dashboard.py ...")
    patch_dashboard()

    print("\n[4/4] Clearing stale model files ...")
    delete_stale()

    print("\n[5/5] Retraining ...")
    retrain()

    print("\n" + "=" * 55)
    print("  ✅ All done! Run: python enhance4_dashboard.py")
    print("=" * 55)