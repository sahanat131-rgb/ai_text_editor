import pandas as pd
from sklearn.model_selection import train_test_split

# Load human data (label 0)
human_df = pd.read_csv("data/human.csv")
human_df = human_df[["text"]].dropna()
human_df["label"] = 0

# Load AI data (label 1)
ai_df = pd.read_csv("data/ai.csv")
ai_df = ai_df[["text"]].dropna()
ai_df["label"] = 1

print(f"Human samples: {len(human_df)}")
print(f"AI samples: {len(ai_df)}")

# Balance: use the smaller count so classes are equal
n = min(len(human_df), len(ai_df))
human_df = human_df.sample(n=n, random_state=42)
ai_df = ai_df.sample(n=n, random_state=42)

# Combine and shuffle
combined = pd.concat([human_df, ai_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Combined total: {len(combined)}")
print(combined["label"].value_counts())

# Split: 70% train, 15% val, 15% test
train, temp = train_test_split(combined, test_size=0.3, random_state=42, stratify=combined["label"])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print("Saved train.csv, val.csv, test.csv to data/")