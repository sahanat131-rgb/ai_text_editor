import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MAX_CHARS = 300
TWEET_WEIGHT = 30  # each of your tweet rows counts as 30 rows, so the small set is not drowned out

# 1. Abstracts (large dataset)
df = pd.read_csv("data/ai_human_train.csv")
df["text"] = df["abstract"].astype(str).str.strip().str[:MAX_CHARS]
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset="text")
df = df[df["text"].str.len() > 20][["text", "label"]]
abs_train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
abs_val, abs_test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

# 2. Your tweets (small dataset): train part for training, val + test kept for evaluation only
tw_train = pd.read_csv("data/train.csv")[["text", "label"]]
tw_eval = pd.concat([pd.read_csv("data/val.csv"), pd.read_csv("data/test.csv")])[["text", "label"]]

# 3. Train v3 on both, giving the tweets extra weight
train = pd.concat([abs_train, tw_train])
weights = np.concatenate([np.ones(len(abs_train)), np.full(len(tw_train), float(TWEET_WEIGHT))])
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
model = LogisticRegression(max_iter=1000)
model.fit(vec.fit_transform(train["text"].astype(str)), train["label"], sample_weight=weights)


def evaluate(m, v, data):
    X = v.transform(data["text"].astype(str))
    pred = m.predict(X)
    conf = m.predict_proba(X).max(axis=1)
    return {
        "acc": round(accuracy_score(data["label"], pred), 3),
        "f1": round(f1_score(data["label"], pred), 3),
        "conf": round(float(conf.mean()), 3),
    }


models = {
    "v1": (joblib.load("model.pkl"), joblib.load("vectorizer.pkl")),
    "v2": (joblib.load("model_v2.pkl"), joblib.load("vectorizer_v2.pkl")),
    "v3": (model, vec),
}
rows = {}
for name, (m, v) in models.items():
    rows[f"{name} on abstracts"] = evaluate(m, v, abs_test)
    rows[f"{name} on tweets"] = evaluate(m, v, tw_eval)

print(f"\ntweet rows used only for evaluation: {len(tw_eval)}")
print(f"{'':22}{'accuracy':>10}{'f1':>8}{'avg confidence':>16}")
for name, r in rows.items():
    print(f"{name:22}{r['acc']:>10}{r['f1']:>8}{r['conf']:>16}")

# 4. Promotion gate: v3 must beat v1 on tweets AND on abstracts
v3_tw, v1_tw = rows["v3 on tweets"], rows["v1 on tweets"]
v3_ab, v1_ab = rows["v3 on abstracts"], rows["v1 on abstracts"]
promote = v3_tw["f1"] > v1_tw["f1"] and v3_ab["f1"] > v1_ab["f1"]
print("\nPROMOTION GATE:", "PASSED - v3 can replace v1" if promote else "FAILED - keep v1 live")

# 5. Log to MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ai-text-detector")
with mlflow.start_run(run_name="v3-mixed"):
    mlflow.log_params({"tweet_weight": TWEET_WEIGHT, "max_chars": MAX_CHARS, "train_rows": len(train)})
    mlflow.log_metrics({
        "abstracts_f1": v3_ab["f1"],
        "tweets_f1": v3_tw["f1"],
        "tweets_conf": v3_tw["conf"],
        "promoted": float(promote),
    })

joblib.dump(model, "model_v3.pkl")
joblib.dump(vec, "vectorizer_v3.pkl")
print("Saved model_v3.pkl and vectorizer_v3.pkl")