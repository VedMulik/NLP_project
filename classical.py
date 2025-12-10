# classical_baseline.py

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from load_data import load_qevasion

def train_classical_baseline():
    train_df, test_df = load_qevasion()
    
    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(int)

    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(int)

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1,2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced"
    )

    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)

    print("\n=== Classical ML Baseline (TF-IDF + Logistic Regression) ===")
    print(classification_report(y_test, preds, digits=4))

 
    joblib.dump(clf, "classical_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Saved model and vectorizer.")

if __name__ == "__main__":
    train_classical_baseline()
