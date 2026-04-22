# spam-classifier
Spam Classifier (Machine Learning)  A machine learning-based Spam Classifier that automatically detects whether a message is spam or not spam (ham) using Natural Language Processing (NLP) techniques.  This project focuses on text preprocessing, feature extraction, and classification models to build an efficient and accurate spam detection system.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.7+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and TF-IDF feature extraction.

---

## 📌 Problem Statement

Spam messages are a major nuisance and security threat. This project builds a classifier that can automatically detect spam SMS messages with high accuracy, using classical ML techniques and NLP preprocessing.

---

## 📂 Dataset

- **Source:** [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 SMS messages
- **Labels:** `spam` (747) and `ham` (4,825)
- **Note:** Dataset is imbalanced (~87% ham, ~13% spam) — handled via evaluation metrics

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Pandas | Data loading & manipulation |
| NLTK | Text preprocessing & stopwords |
| Scikit-learn | TF-IDF, ML models, evaluation |
| Matplotlib & Seaborn | Visualization |
| Pickle | Model serialization |

---

## 🔄 Project Pipeline

```
Raw Text → Preprocessing → TF-IDF Vectorization → Model Training → Evaluation → Prediction
```

### Phase 1: Data Loading & EDA
- Loaded dataset with `pandas`
- Inspected class distribution
- Visualized spam vs ham imbalance

### Phase 2: Text Preprocessing
- Converted text to lowercase
- Removed punctuation and special characters
- Removed English stopwords using NLTK
- Encoded labels: `spam → 1`, `ham → 0`

### Phase 3: Feature Engineering
- Applied **TF-IDF Vectorization** with `max_features=3000`
- TF-IDF captures word importance across all messages
- Converted text into numerical feature vectors of shape `(5572, 3000)`

### Phase 4: Model Training
- Split data: **80% train / 20% test**
- Trained two models:
  - **Multinomial Naive Bayes**
  - **Logistic Regression**

### Phase 5: Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization

### Phase 6: Prediction Function
- Built a reusable `predict_message()` function for custom inputs

### Phase 7: Model Saving
- Saved model and vectorizer using `pickle` for deployment

---

## 📊 Results

### Naive Bayes
| Metric | Ham | Spam |
|--------|-----|------|
| Precision | 0.97 | **1.00** |
| Recall | 1.00 | 0.81 |
| F1-Score | 0.99 | 0.90 |
| **Accuracy** | | **97.4%** |

### Logistic Regression
| Metric | Ham | Spam |
|--------|-----|------|
| Precision | 0.95 | 0.96 |
| Recall | 1.00 | 0.64 |
| F1-Score | 0.97 | 0.77 |
| **Accuracy** | | **94.8%** |

### 🏆 Winner: Naive Bayes
- Higher accuracy (97.4% vs 94.8%)
- **Zero False Positives** — no ham ever wrongly marked as spam
- Better spam F1-Score (0.90 vs 0.77)

---

## 🔍 Confusion Matrix Analysis

### Naive Bayes
```
                Predicted Ham    Predicted Spam
Actual Ham           965               0
Actual Spam           28             122
```

### Logistic Regression
```
                Predicted Ham    Predicted Spam
Actual Ham           961               4
Actual Spam           54              96
```

> **Key Insight:** In spam classification, False Negatives (spam reaching inbox) are more dangerous than False Positives (ham going to spam). Naive Bayes achieves fewer False Negatives (28 vs 54).

---

## 💡 Key Design Decisions

- **Why TF-IDF over Bag of Words?** TF-IDF penalizes common words and rewards rare but meaningful words — better signal for spam detection.
- **Why keep numbers in preprocessing?** Phone numbers and prize amounts (e.g., `£1000`, `08001234`) are strong spam indicators.
- **Why Naive Bayes for text?** Fast, works well with high-dimensional sparse data, and naturally suited for word probability calculations.
- **Why not just use accuracy?** Dataset is imbalanced — a model predicting all ham gets 87% accuracy. Precision, Recall and F1 give a truer picture.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier

# 2. Install dependencies
pip install pandas scikit-learn nltk matplotlib seaborn

# 3. Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# 4. Run the notebook
jupyter notebook spam_classifier.ipynb
```

---

## 🔮 Predict Custom Messages

```python
import pickle
from your_module import clean_text

# Load saved model
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def predict_message(message):
    cleaned = clean_text(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "🚨 SPAM" if prediction == 1 else "✅ HAM"

# Test it!
print(predict_message("Congratulations! You won a free prize!"))  # 🚨 SPAM
print(predict_message("Hey, are we meeting for lunch tomorrow?")) # ✅ HAM
```

---

## 📁 Project Structure

```
spam-classifier/
│
├── spam.csv                    # Dataset
├── spam_classifier.ipynb       # Main notebook
├── spam_classifier.pkl         # Saved Naive Bayes model
├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
└── README.md                   # This file
```

---

## 🔭 Future Work

- [ ] Extend to email spam using the **Enron Email Dataset**
- [ ] Experiment with **Random Forest** and **SVM** models
- [ ] Improve spam recall using **class_weight balancing**
- [ ] Try **deep learning** approach with LSTM

---

## 🧠 Learnings

- Handling class imbalance in NLP classification
- Importance of choosing the right evaluation metric
- How TF-IDF captures meaningful word signals
- Trade-off between Precision and Recall in real-world classifiers
- Model serialization for deployment readiness

---

