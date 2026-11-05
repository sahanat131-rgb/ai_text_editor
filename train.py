import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading data...")
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

X_train_text = train_df["text"]
y_train = train_df["label"]
X_val_text = val_df["text"]
y_val = val_df["label"]

print(f"Train size: {len(X_train_text)}, Val size: {len(X_val_text)}")

mlflow.set_experiment("ai-text-detector")

with mlflow.start_run():
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}")

    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved model.pkl and vectorizer.pkl")

print("Done.")