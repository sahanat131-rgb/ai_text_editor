import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "https://ai-text-editor-jvzw.onrender.com/predict"
LOG_FILE = Path("predictions_log.csv")
TRAIN_FILE = Path("data/train.csv")
COLUMNS = ["timestamp", "text", "text_length", "prediction", "confidence", "latency_s"]

SAMPLES = [
    "@united my flight was delayed again and nobody at the gate could tell us anything",
    "Thanks for the quick help with my baggage, great service today!",
    "The committee will convene next Tuesday to review the proposed amendments to the quarterly budget.",
    "In conclusion, artificial intelligence offers numerous benefits across diverse industries, enhancing efficiency and innovation.",
    "I need an entire movie on rukmini and krishna, these few minutes were so divine",
    "lol that match last night was insane, can't believe he scored in the 90th minute",
]

st.set_page_config(page_title="AI Text Detector Monitor", layout="wide")
st.title("AI Text Detector - Monitoring Dashboard")
st.caption("Sends texts to the live API, logs every prediction, and watches for low confidence and drift.")


def load_log():
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=COLUMNS)


def predict_and_log(text):
    start = time.time()
    r = requests.post(API_URL, json={"text": text}, timeout=90)
    r.raise_for_status()
    out = r.json()
    row = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "text": text,
        "text_length": len(text),
        "prediction": out["prediction"],
        "confidence": out["confidence"],
        "latency_s": round(time.time() - start, 2),
    }
    df = pd.concat([load_log(), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    return row


left, right = st.columns(2)

with left:
    st.subheader("Try a text")
    text = st.text_area("Text to check", height=120)
    if st.button("Predict") and text.strip():
        try:
            with st.spinner("Calling the API (the first call can take ~50s if it was asleep)..."):
                row = predict_and_log(text.strip())
            st.success(f"{row['prediction']}  (confidence {row['confidence']})")
        except requests.exceptions.RequestException as e:
            st.error(f"API call failed: {e}")

with right:
    st.subheader("Send a sample batch")
    st.write("Sends 6 texts from different styles (tweets, formal, AI-like, devotional, sports).")
    if st.button("Run sample batch"):
        try:
            with st.spinner("Sending samples..."):
                for s in SAMPLES:
                    predict_and_log(s)
            st.success("Done. Scroll down to see the results.")
        except requests.exceptions.RequestException as e:
            st.error(f"API call failed: {e}")

log = load_log()
st.divider()

if log.empty:
    st.info("No predictions logged yet. Use the buttons above.")
    st.stop()

st.subheader("Monitoring")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total predictions", len(log))
m2.metric("% predicted AI", f"{(log['prediction'] == 'AI-generated').mean() * 100:.0f}%")
m3.metric("Average confidence", f"{log['confidence'].mean():.2f}")
m4.metric("Average latency (s)", f"{log['latency_s'].mean():.1f}")

c1, c2 = st.columns(2)
with c1:
    st.write("Confidence distribution")
    bins = pd.cut(log["confidence"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    st.bar_chart(bins.value_counts().sort_index().rename(lambda i: str(i)))
with c2:
    st.write("Predictions by label")
    st.bar_chart(log["prediction"].value_counts())

st.subheader("Drift checks")
low_share = (log["confidence"] < 0.6).mean()
if low_share > 0.5:
    st.warning(f"{low_share * 100:.0f}% of predictions have confidence below 0.6. The model is unsure on most inputs.")
else:
    st.success(f"Only {low_share * 100:.0f}% of predictions have confidence below 0.6.")

if TRAIN_FILE.exists():
    train_len = pd.read_csv(TRAIN_FILE)["text"].astype(str).str.len().mean()
    live_len = log["text_length"].mean()
    change = (live_len - train_len) / train_len * 100
    st.write(f"Average text length: training {train_len:.0f} chars vs live {live_len:.0f} chars ({change:+.0f}%)")
    if abs(change) > 30:
        st.warning("Live text length differs from the training data by more than 30%. Possible input drift.")
    else:
        st.success("Live text length is close to the training data.")
else:
    st.info("data/train.csv not found, so the length comparison is skipped.")

st.subheader("Prediction log")
st.dataframe(log.sort_values("timestamp", ascending=False), width="stretch")