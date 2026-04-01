# 🚀 SpamGuard-Intelligent Spam Detection & Spammer Identification System

### 💡 Machine Learning + Cloud-Based Real-Time Detection

---

## 📌 Overview

This project presents an **intelligent machine learning system** for detecting spam messages and identifying spammers in **industrial mobile cloud environments**.

The system integrates:

* Content-based spam detection
* Behavioral analysis
* Ensemble machine learning models
* Real-time cloud deployment

Unlike traditional systems, this project supports **continuous data streams and real-time detection**, addressing limitations of batch-based approaches .

---

## 🎯 Objectives

* Detect spam messages with high accuracy
* Identify spammers using behavioral patterns
* Support real-time cloud-based detection
* Handle multiaspect data (text + behavior)
* Reduce false positives and improve reliability

---

## 🏗️ System Architecture

The system consists of 4 major layers:

1. **Data Layer** – Dataset collection & preprocessing
2. **ML Layer** – Feature extraction + model training
3. **Enhanced Layer** – Behavioral + ensemble models
4. **Cloud Layer** – Real-time detection & dashboard

---

# 📂 COMPLETE PROJECT STRUCTURE

```bash id="proj-structure"
SDITS/
│
├── spamguard-deploy/                     # 🔥 FINAL DEPLOYABLE SYSTEM
│   │
│   ├── __pycache__/                     # Compiled Python files
│   │
│   ├── data/                            # Dataset
│   │   └── cleaned_data.csv
│   │
│   ├── templates/                       # Frontend UI
│   │   └── dashboard.html
│   │
│   ├── cloud_app.py                     # Main Flask cloud server
│   ├── device_simulator.py              # Simulates real-time data
│   ├── enhance1_behavioral_features.py  # Behavioral features
│   ├── enhance2_combined_model.py       # Combined ML model
│   ├── start_cloud.py                   # Start cloud locally
│   ├── train_on_render.py               # Training for deployment
│   ├── render.yaml                      # Cloud deployment config
│   ├── requirements.txt                 # Dependencies
│   └── README.md
│
├── spammer_cloud/                       # ☁️ CLOUD MODEL (PRODUCTION)
│   │
│   ├── __pycache__/
│   │
│   ├── data/
│   │   ├── cleaned_data.csv
│   │   └── combined_model.pkl           # Trained model
│   │
│   ├── templates/
│   │   └── dashboard.html
│   │
│   ├── cloud_app.py                     # Flask app (production)
│   ├── device_simulator.py              # Cloud testing simulator
│   ├── enhance1_behavioral_features.py  # Feature engineering
│   ├── enhance2_combined_model.py       # Model logic
│   ├── start_cloud.py                   # Run cloud
│   ├── render.yaml                      # Deployment config
│   ├── requirements.txt
│   └── README.md
│
├── spammer_detection/                   # 🧠 BASE ML PIPELINE
│   │
│   ├── data/
│   │   ├── cleaned_data.csv
│   │   ├── features.pkl
│   │   ├── SMSSpamCollection
│   │   ├── smsspamcollection.zip
│   │   └── trained_models.pkl
│   │
│   ├── results/
│   │   ├── confusion_matrix.png
│   │   └── performance_comparison.png
│   │
│   ├── step1_get_dataset.py             # Load dataset
│   ├── step2_preprocess.py              # Cleaning & preprocessing
│   ├── step3_feature_extraction.py      # TF-IDF, NLP features
│   ├── step4_train_models.py            # Train ML models
│   ├── step5_visualize_results.py       # Graphs & analysis
│   ├── step6_predict.py                 # Prediction script
│   └── README.md
│
├── spammer_enhanced/                    # ⚡ ADVANCED SYSTEM
│   │
│   ├── __pycache__/
│   │
│   ├── data/
│   │   ├── cleaned_data_original.csv
│   │   ├── cleaned_data.csv
│   │   ├── combined_model.pkl
│   │   └── enhanced_features.pkl
│   │
│   ├── results/
│   │   ├── explanation_msg1.png
│   │   ├── explanation_msg2.png
│   │   ├── explanation_msg3.png
│   │   └── feature_importance.png
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── enhance1_behavioral_features.py   # Behavioral analysis
│   ├── enhance2_combined_model.py        # Ensemble model
│   ├── enhance3_explainability.py        # Explainable AI
│   ├── enhance4_dashboard.py             # Dashboard
│   ├── patch_and_retrain.py              # Model updates
│   ├── run_all.py                        # Run entire pipeline
│   └── README.md
│
├── pylint/                              # Code quality checks
│
└── requirements.txt                     # Global dependencies
```

---

## ⚙️ Technologies Used

* Python
* Scikit-learn
* Pandas, NumPy
* Flask (Web + Cloud)
* HTML/CSS (Dashboard UI)
* NLP (TF-IDF, text processing)

---

## 🤖 Machine Learning Models

* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* Ensemble Learning (Combined Model)

ML models outperform traditional rule-based systems in spam detection tasks .

---

## 🔍 Features

✔ Spam Detection (SMS / Email / Messages)
✔ Spammer Identification (Behavior-based)
✔ Real-Time Cloud Detection
✔ Device Simulator for Testing
✔ Explainable AI (Feature importance, explanations)
✔ Dashboard Visualization
✔ Ensemble Model for Higher Accuracy

---

## 🚀 How to Run

### 1️⃣ Install Requirements

```bash id="run1"
pip install -r requirements.txt
```

---

### 2️⃣ Run Base Model

```bash id="run2"
cd spammer_detection
python step1_get_dataset.py
python step2_preprocess.py
python step3_feature_extraction.py
python step4_train_models.py
```

---

### 3️⃣ Run Enhanced Model

```bash id="run3"
cd spammer_enhanced
python run_all.py
```

---

### 4️⃣ Run Cloud System (Main)

```bash id="run4"
cd spamguard-deploy
python start_cloud.py
```

Open:

```
http://localhost:5001
```

---

### 5️⃣ Run Device Simulator

```bash id="run5"
python device_simulator.py
```

---

## 📊 Performance Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📈 Results

* High spam detection accuracy
* Reduced false positives
* Improved performance using combined features
* Real-time detection capability

---

## 🔐 Applications

* Industrial Mobile Cloud Systems
* IoT Communication Security
* SMS & Email Filtering
* Social Media Platforms

---

## 🚧 Challenges Addressed

* Dynamic spam patterns
* Data imbalance
* Real-time processing
* High-dimensional feature handling

---

## 🔮 Future Work

* Deep Learning (LSTM, BERT)
* Multilingual spam detection
* Edge computing integration
* Real-time streaming analytics

---

## 👨‍💻 Authors

* Rathindhar R M
* Santhosh P
* Prakash P

---

## 📜 License

For academic and research use only.

---

## ⭐ Acknowledgement

This project is inspired by multiple IEEE research works on spam detection, machine learning, IoT security, and cloud-based intrusion detection systems.
