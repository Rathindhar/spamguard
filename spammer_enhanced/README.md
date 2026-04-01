# SpamGuard — Enhanced Spammer Detection System
### Full Enhanced Version | IEEE Paper Implementation

---

## 🆕 What's New vs Basic Version

| Feature | Basic Version | Enhanced Version |
|---------|--------------|-----------------|
| Features | TF-IDF only | TF-IDF + 20 Behavioral features |
| Model | 4 individual classifiers | Stacked Ensemble (meta-learner) |
| Accuracy | ~97.1% (SVM) | Higher (combined features) |
| Explainability | None | Per-message + global XAI charts |
| Drift Detection | None | Page-Hinkley drift monitor |
| Interface | Terminal only | Full web dashboard |
| API | None | REST API (/api/predict) |
| Batch analysis | None | Analyse many messages at once |

---

## 📁 File Structure

```
spammer_enhanced/
├── run_all.py                      ← Run everything at once
│
├── enhance1_behavioral_features.py ← 20 behavioral features
├── enhance2_combined_model.py      ← Stacked ensemble + drift
├── enhance3_explainability.py      ← XAI charts
├── enhance4_dashboard.py           ← Flask web app
│
├── templates/
│   └── index.html                  ← Dashboard UI
│
├── data/                           ← Auto-created
│   ├── cleaned_data.csv            ← From basic step2
│   ├── enhanced_features.pkl       ← Behavioral features
│   └── combined_model.pkl          ← Trained ensemble
│
└── results/                        ← Auto-created
    ├── feature_importance.png      ← Global XAI chart
    ├── explanation_msg1.png        ← Per-message explanations
    ├── explanation_msg2.png
    └── explanation_msg3.png
```

---

## 🚀 How to Run

### Option A — Run everything at once (recommended)
```bash
python run_all.py
```

### Option B — Step by step
```bash
# Prerequisites: run basic version steps 1-3 first
python enhance1_behavioral_features.py
python enhance2_combined_model.py
python enhance3_explainability.py
python enhance4_dashboard.py          # launches at localhost:5000
```

---

## 🌐 Web Dashboard Features

After launching, open **http://localhost:5000**

- **Single Message Analysis** — paste any message, get instant verdict
  with spam probability bar, top triggering words, and behavioral signals
- **Batch Analysis** — paste 10+ messages, get a summary of spam vs legitimate
- **Quick Test Examples** — pre-loaded spam and legitimate examples
- **Drift Detector Panel** — live model health monitoring
- **REST API** — integrate with other systems:

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "Congratulations! You won a free iPhone!"}'
```

---

## 🔬 Enhancement Details

### 1. Behavioral Features (20 signals)
Captures HOW spammers write, not just WHAT they say:
- Uppercase ratio, exclamation count, URL count
- Spam keyword score, currency symbols, repeated characters
- Lexical diversity, sentence structure, action verb detection
- ... and 11 more

### 2. Stacked Ensemble
- Level 0: NB + LR + RF + SVM each make predictions
- Level 1: Meta-learner combines their outputs optimally
- More robust than any single classifier alone

### 3. Concept Drift Detection (Page-Hinkley)
- Monitors prediction accuracy over time
- Alerts when spam patterns have evolved
- Signals when model retraining is needed

### 4. Explainability (XAI)
- Global: which features matter most overall
- Per-message: exactly which words/behaviors triggered the verdict

---

## 📦 Requirements

```bash
pip install flask scikit-learn pandas numpy matplotlib scipy
```

Python 3.8+ recommended.

---

## 🔗 Paper Mapping

| Enhancement | Paper Section |
|-------------|--------------|
| Behavioral features | Section VI Future Work — "temporal and behavioral patterns" |
| Drift detection | Section II-D — "concept drift" challenge |
| Stacked ensemble | Section V — improving over individual classifiers |
| XAI explainability | Section VI Future Work — industrial auditability |
| Web dashboard | Section III-A — "cloud-based infrastructure" deployment |
