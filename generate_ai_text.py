print("SCRIPT STARTED")

import requests
import pandas as pd
import time
import csv
import os
import traceback

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
MODEL = "liquid/lfm-2.5-2.6b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
NUM_SAMPLES = 438
OUTPUT_FILE = "data/ai.csv"
HUMAN_FILE = "data/human.csv"

headers = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
}

def query(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.9,
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        except requests.exceptions.RequestException as e:
            wait = 10 * (attempt + 1)
            print(f"Connection error ({e}), waiting {wait}s...")
            time.sleep(wait)
            continue
        print("QUERY: got status", r.status_code)
        if r.status_code in (400, 401, 402, 403, 404):
            raise SystemExit("Stopping: fix the error above, then rerun.")
        return None
    return None

def main():
    print("MAIN: loading human.csv")
    human_df = pd.read_csv(HUMAN_FILE)
    print("MAIN: loaded", len(human_df), "rows")
    topics = human_df["text"].dropna().sample(n=100, random_state=42).tolist()
    print("MAIN: got", len(topics), "topics")

    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_FILE)

    print("MAIN: opening output file, starting loop")
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "label"])

        generated = 0
        i = 0
        while generated < NUM_SAMPLES:
            print(f"LOOP: i={i}, generated={generated}")
            topic = topics[i % len(topics)]
            prompt = (
                "Write one short, natural social media post (1-3 sentences) "
                f"on a similar topic to this tweet: \"{topic[:150]}\". "
                "Output only the post, and don't mention being an AI."
            )
            text = query(prompt)
            print("LOOP: query returned:", repr(text)[:80])
            if text:
                writer.writerow([text, 1])
                f.flush()
                generated += 1
                print(f"Generated {generated}/{NUM_SAMPLES}")
            i += 1
            time.sleep(2)

    print(f"Done. Saved {generated} AI samples to {OUTPUT_FILE}")

print("SCRIPT: about to call main")
try:
    main()
except Exception as e:
    print("CRASHED WITH EXCEPTION:")
    traceback.print_exc()
print("SCRIPT: finished")