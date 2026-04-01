# Spammer Identification in Industrial Mobile Cloud Systems
### Based on the IEEE Paper — Full Python Implementation

---

## 📁 Project Structure

```
spammer_detection/
│
├── step1_get_dataset.py       ← Download the SMS Spam dataset
├── step2_preprocess.py        ← Clean & normalise the messages
├── step3_feature_extraction.py← Convert text to TF-IDF numbers
├── step4_train_models.py      ← Train all 4 ML classifiers
├── step5_visualize_results.py ← Generate charts & confusion matrix
├── step6_predict.py           ← Predict on new/live messages
│
├── data/                      ← Created automatically
│   ├── SMSSpamCollection      ← Raw dataset (after step 1)
│   ├── cleaned_data.csv       ← Cleaned text (after step 2)
│   ├── features.pkl           ← TF-IDF features (after step 3)
│   └── trained_models.pkl     ← Saved models (after step 4)
│
└── results/                   ← Created automatically
    ├── performance_comparison.png  ← Bar chart (matches Table I)
    └── confusion_matrix.png        ← Confusion matrix (Table II)
```

---

## 🚀 How to Run (Step by Step)

Open a terminal / command prompt in this folder and run each
script in order:

### Step 1 — Download Dataset
```bash
python step1_get_dataset.py
```
Downloads the SMS Spam Collection dataset (~500KB).
> If the download fails, visit https://archive.ics.uci.edu/dataset/228/sms+spam+collection
> and manually place `SMSSpamCollection` in the `data/` folder.

---

### Step 2 — Pre-process Data
```bash
python step2_preprocess.py
```
Cleans text: lowercase → remove special chars → tokenise → remove stop-words.

---

### Step 3 — Extract Features (TF-IDF)
```bash
python step3_feature_extraction.py
```
Converts messages into numerical vectors using TF-IDF.
Splits data into 80% training / 20% testing.

---

### Step 4 — Train All 4 ML Models
```bash
python step4_train_models.py
```
Trains and evaluates:
- Naive Bayes
- Logistic Regression
- Random Forest
- Support Vector Machine

Prints accuracy, precision, recall, and F1-score for each.

---

### Step 5 — Visualise Results
```bash
python step5_visualize_results.py
```
Saves two charts in the `results/` folder:
- `performance_comparison.png` — matches Table I in the paper
- `confusion_matrix.png` — matches Table II in the paper

---

### Step 6 — Predict New Messages
```bash
python step6_predict.py
```
Tests the best model on example spam/ham messages, then
lets you type your own messages for live prediction.

---

## 📦 Requirements

Install dependencies with:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

Python 3.8 or newer is recommended.

---

## 🔗 Dataset

**SMS Spam Collection Dataset**
- Source: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Size: 5,574 messages (4,827 ham + 747 spam)
- License: Public domain / research use

---

## 📄 Reference

This implementation is based on:
> "An Intelligent Machine Learning Model for Spammer Identification
> in Industrial Mobile Cloud Systems"
> Panimalar Engineering College, Chennai, India

Key paper sections implemented:
- Section III-C : Proposed Methodology (6-step pipeline)
- Section IV-B  : Data Preprocessing
- Section IV-C  : Feature Extraction (TF-IDF)
- Section IV-D  : Machine Learning Model Training
- Section V     : Results and Discussion
