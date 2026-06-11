# SMS Spam Detection using Machine Learning and Deep Learning

## Project Overview

This project implements an end-to-end SMS Spam Detection system using Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques.

The goal is to classify SMS messages into two categories:

* Ham (Legitimate Message)
* Spam (Unwanted Promotional or Fraudulent Message)

The project covers the complete machine learning pipeline including:

* Exploratory Data Analysis (EDA)
* Text Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Visualization
* Prediction Pipeline
* Deep Learning with RNN

---

# Dataset

Dataset: SMS Spam Collection Dataset

The dataset contains SMS messages labeled as either:

| Label | Description    |
| ----- | -------------- |
| ham   | Normal message |
| spam  | Spam message   |

After preprocessing:

| Label | Encoded Value |
| ----- | ------------- |
| ham   | 0             |
| spam  | 1             |

---

# Project Structure

```text
spam-detection/
│
├── dataset/
│   ├── spam.csv
│   ├── spam_processed.csv
│   └── model_comparison.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── rnn_tokenizer.pkl
│   └── rnn_model.keras
│
├── notebooks/
│   ├── eda.ipynb
│   ├── nlp.ipynb
│   ├── feature_engineering.ipynb
│   ├── classical_ml.ipynb
│   ├── visualization.ipynb
│   ├── final_demo.ipynb
│   └── deep_learning.ipynb
│
├── src/
│   ├── eda_preprocessing.py
│   ├── nlp_preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluation.py
│   └── prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Exploratory Data Analysis (EDA)

The following preprocessing tasks were performed:

* Dataset inspection
* Shape analysis
* Missing value detection
* Duplicate record detection
* Duplicate removal
* Column cleaning and renaming
* Label encoding

Operations:

* Removed unnecessary columns
* Removed duplicate messages
* Renamed columns:

  * label
  * message
* Converted:

  * ham → 0
  * spam → 1

---

# NLP Preprocessing

The following NLP preprocessing techniques were applied:

## 1. Lowercasing

Converts all text into lowercase.

Example:

```text
HELLO WORLD
```

↓

```text
hello world
```

## 2. Remove Punctuation

Example:

```text
Hello!!!
```

↓

```text
Hello
```

## 3. Remove Special Characters and Numbers

Example:

```text
Win $1000 now
```

↓

```text
Win now
```

## 4. Tokenization

Example:

```text
hello world
```

↓

```python
['hello', 'world']
```

## 5. Stopword Removal

Example:

```python
['this', 'is', 'a', 'message']
```

↓

```python
['message']
```

## 6. Stemming

Example:

```text
running
runs
runner
```

↓

```text
run
run
runner
```

---

# Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) was used to transform text into numerical vectors.

Why TF-IDF?

* Captures word importance
* Reduces influence of common words
* Works extremely well for text classification

Configuration:

```python
TfidfVectorizer(max_features=5000)
```

Output:

```text
Text
 ↓
TF-IDF
 ↓
Numerical Feature Matrix
```

---

# Train / Validation / Test Split

The dataset was divided into:

| Dataset        | Percentage |
| -------------- | ---------- |
| Training Set   | 70%        |
| Validation Set | 15%        |
| Test Set       | 15%        |

Stratified sampling was used to preserve class distribution.

---

# Machine Learning Models

Three machine learning models were trained and evaluated.

## 1. Multinomial Naive Bayes

Advantages:

* Fast
* Effective for text classification
* Strong baseline model

---

## 2. Logistic Regression

Advantages:

* Simple
* Interpretable
* Strong linear classifier

---

## 3. Random Forest

Advantages:

* Handles complex patterns
* Reduces overfitting
* High predictive performance

---

# Deep Learning Model

## Recurrent Neural Network (RNN)

Architecture:

```text
Embedding Layer
      ↓
SimpleRNN
      ↓
Dense Layer
      ↓
Output Layer
```

Techniques used:

* Tokenization
* Sequence Encoding
* Padding
* Class Weight Balancing

The dataset was imbalanced, therefore class weights were used during training.

---

# Model Evaluation Metrics

The following metrics were used:

## Accuracy

Measures overall prediction correctness.

## Precision

Measures how many predicted spam messages were actually spam.

## Recall

Measures how many actual spam messages were detected.

## F1 Score

Harmonic mean of Precision and Recall.

This metric was used as the primary comparison metric.

---

# Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Naive Bayes         |   96.65% |   100.00% | 73.47% |   84.71% |
| Logistic Regression |   95.10% |    98.39% | 62.24% |   76.25% |
| Random Forest       |   97.16% |    98.72% | 78.57% |   87.50% |
| RNN                 |   96.01% |    83.84% | 84.69% |   84.26% |

---

# Best Performing Model

Random Forest achieved the highest overall performance.

Performance:

* Accuracy: 97.16%
* Precision: 98.72%
* Recall: 78.57%
* F1 Score: 87.50%

Therefore, Random Forest was selected as the final model.

---

# Prediction Pipeline

Prediction Flow:

```text
Raw SMS Message
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Random Forest Model
        ↓
Spam / Ham Prediction
```

Example:

Input:

```text
URGENT! You have won a free lottery prize.
```

Output:

```python
{
    "prediction": "Spam",
    "confidence": 0.74
}
```

---

# Visualization

The following visualizations were created:

* Accuracy Comparison
* Precision Comparison
* Recall Comparison
* F1 Score Comparison
* Confusion Matrix

These visualizations help compare model performance and identify the best-performing model.

---

# Future Improvements

Possible improvements include:

* BERT Fine-Tuning
* Hyperparameter Optimization
* Cross Validation
* Model Deployment using FastAPI
* Web Application Integration
* Real-Time SMS Classification API

---

# Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* TensorFlow
* Matplotlib
* Seaborn
* Pickle

