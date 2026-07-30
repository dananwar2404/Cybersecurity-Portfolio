#Spam Detection Using Machine Learning

> A machine-learning project that classifies messages as spam or legitimate using text-processing and classification techniques.

---

## Overview

This project explores how machine-learning algorithms can be used to detect spam messages. The workflow includes preparing the dataset, cleaning and transforming text, training classification models, evaluating model performance, and reviewing the final predictions.

---

## Technologies

- Python
- Jupyter Notebook
- Pandas
- Scikit-learn
- Natural Language Processing
- Machine Learning

---

## Project Files

| File | Description |
|------|-------------|
| 📓 [Machine Learning Notebook](./Lab2_Spam_Detection.ipynb) | Complete Jupyter notebook containing the data preprocessing, feature engineering, model training, and evaluation workflow. |
| 📊 [Confusion Matrix](./confusion-matrix.png) | Visualization of the model's classification performance. |
| 📄 [Reflection Paper](./Reflection.docx) | Written analysis discussing the project methodology, results, and lessons learned. |

 **Python Source Code**
<div class="terminal-viewer">

<div class="terminal-header">
<span class="terminal-dot"></span>
<span class="terminal-dot"></span>
<span class="terminal-dot"></span>
<span class="terminal-title">Lab2_Spam_Detection.py</span>
</div>

```python
#!/usr/bin/env python
# coding: utf-8

# Part 1: Data Loading and Exploration

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(
    r"C:\AIcybersec\spam.csv.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("Dataset shape:", df.shape)
print("\nFirst 5 messages:")
print(df.head())

print("\nClass distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

print("\nMissing values:")
print(df.isnull().sum())


# Part 2: Text Preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Convert labels to binary
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"Training set: {len(X_train)} messages")
print(f"Test set: {len(X_test)} messages")

# Convert text into numerical features
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)} unique words")
print(f"Training matrix shape: {X_train_counts.shape}")
print(f"Test matrix shape: {X_test_counts.shape}")


# Part 3: Model Training

from sklearn.naive_bayes import MultinomialNB
import time

clf = MultinomialNB()

start_time = time.time()
clf.fit(X_train_counts, y_train)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.4f} seconds")
print("Model is ready to make predictions!")


# Part 4: Model Evaluation

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

y_pred = clf.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)

print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham (0)", "Spam (1)"]
    )
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Part 5: Testing Custom Messages

test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "WINNER! You've won a $1000 Walmart gift card. Click here NOW!",
    "Your package will arrive tomorrow between 2-4 PM",
    "Congratulations! You have been selected for a FREE cruise. Call now!",
    "Can you send me the homework assignment from class?"
]

test_counts = vectorizer.transform(test_messages)
predictions = clf.predict(test_counts)
probabilities = clf.predict_proba(test_counts)

print("Custom Message Predictions:")
print("=" * 80)

for index, message in enumerate(test_messages):
    label = "SPAM" if predictions[index] == 1 else "HAM"
    spam_probability = probabilities[index][1] * 100

    print(f'\nMessage: "{message}"')
    print(
        f"Prediction: {label} "
        f"(Spam probability: {spam_probability:.2f}%)"
    )


# Part 6: TF-IDF Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf_tfidf = MultinomialNB()
clf_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

print(
    f"TF-IDF Accuracy: "
    f"{accuracy_tfidf:.4f} ({accuracy_tfidf * 100:.2f}%)"
)

print(
    f"CountVectorizer Accuracy: "
    f"{accuracy:.4f} ({accuracy * 100:.2f}%)"
)


# Part 7: Model Comparison

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

print("Model Comparison Results")
print("=" * 60)

for model_name, model in models.items():
    model.fit(X_train_counts, y_train)
    model_predictions = model.predict(X_test_counts)
    model_accuracy = accuracy_score(y_test, model_predictions)

    print(
        f"{model_name}: "
        f"{model_accuracy:.4f} ({model_accuracy * 100:.2f}%)"
    )


# Part 8: Cross-Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    clf,
    X_train_counts,
    y_train,
    cv=5
)

print(f"Cross-validation scores: {scores}")
print(
    f"Mean accuracy: "
    f"{scores.mean():.4f} (+/- {scores.std():.4f})"
)
```

</div>

## Methodology

1. Imported and reviewed the message dataset
2. Cleaned and prepared the text data
3. Converted text into numerical features
4. Trained a spam-classification model
5. Tested the model using unseen data
6. Evaluated performance using a confusion matrix

---

## Model Evaluation

![Confusion Matrix](./Confusion%20matrix.png)

The confusion matrix shows how accurately the model classified spam and legitimate messages. It also helps identify false positives and false negatives.

---

## Skills Demonstrated

- Data preprocessing
- Text classification
- Machine-learning model development
- Natural language processing
- Model evaluation
- Confusion-matrix interpretation
- Python programming
- Technical analysis

---

## What I Learned

This project strengthened my understanding of how machine-learning models process text and identify patterns associated with spam. It also provided practical experience preparing data, training a classification model, and interpreting performance metrics.
