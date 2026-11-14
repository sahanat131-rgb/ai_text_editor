import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MAX_CHARS = 300  # keep texts short, closer to social-media length

# 1. Load and clean the new dataset (label 0 = human, 1 = AI)
df = pd.read_csv("data/ai_human_train.csv")
df["text"] = df["abstract"].astype(str).str.strip().str[:MAX_CHARS]
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset="text")
df = df[df["text"].str.len() > 20]
print("rows:", len(df), "| share labelled AI:", round(df["label"].mean(), 3))

# 2. Split 70 / 15 / 15
train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

# 3. Train
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
model = LogisticRegression(max_iter=1000)
model.fit(vec.fit_transform(train["text"]), train["label"])

# 4. Evaluate v2, and v1 for comparison, on the same data
tweets = pd.read_csv("data/test.csv")  # your old tweets + AI tweets, never used in v2 training


def evaluate(m, v, texts, labels):
    X = v.transform(texts.astype(str))
    pred = m.predict(X)
    conf = m.predict_proba(X).max(axis=1)
    return {
        "acc": round(accuracy_score(labels, pred), 3),
        "f1": round(f1_score(labels, pred), 3),
        "conf": round(float(conf.mean()), 3),
    }


old_model = joblib.load("model.pkl")
old_vec = joblib.load("vectorizer.pkl")

results = {
    "v1 on new test set": evaluate(old_model, old_vec, test["text"], test["label"]),
    "v2 on new test set": evaluate(model, vec, test["text"], test["label"]),
    "v1 on your tweets": evaluate(old_model, old_vec, tweets["text"], tweets["label"]),
    "v2 on your tweets": evaluate(model, vec, tweets["text"], tweets["label"]),
}
val_res = evaluate(model, vec, val["text"], val["label"])

print()
print(f"{'':22}{'accuracy':>10}{'f1':>8}{'avg confidence':>16}")
for name, r in results.items():
    print(f"{name:22}{r['acc']:>10}{r['f1']:>8}{r['conf']:>16}")

# 5. Log to MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ai-text-detector")
with mlflow.start_run(run_name="v2-abstracts"):
    mlflow.log_params({
        "dataset": "Ateeqq/AI-and-Human-Generated-Text (train.csv)",
        "rows": len(df),
        "max_chars": MAX_CHARS,
        "max_features": 20000,
        "ngram_range": "1-2",
    })
    mlflow.log_metrics({
        "val_f1": val_res["f1"],
        "test_f1": results["v2 on new test set"]["f1"],
        "test_conf": results["v2 on new test set"]["conf"],
        "tweets_f1": results["v2 on your tweets"]["f1"],
        "tweets_conf": results["v2 on your tweets"]["conf"],
    })

# 6. Save as v2 (does not overwrite the live model)
joblib.dump(model, "model_v2.pkl")
joblib.dump(vec, "vectorizer_v2.pkl")
print("\nSaved model_v2.pkl and vectorizer_v2.pkl")