# My-Seventh-Repository-
Language Detection Model for Multilingual Text
# ==========================================
# FILE: train.py
# ==========================================

import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset.csv")

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    text = str(text).lower()

    # Remove unwanted symbols but keep language characters
    text = re.sub(r"[^\w\sÀ-ÿء-ي¿¡]", "", text)

    return text

# Apply preprocessing
df["Text"] = df["Text"].apply(preprocess_text)

# ==========================================
# FEATURES & LABELS
# ==========================================

X = df["Text"]
y = df["Language"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# MODEL PIPELINE
# ==========================================

model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL ACCURACY:", accuracy)
print("================================\n")

print("CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")