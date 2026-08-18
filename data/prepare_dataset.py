import pandas as pd
import os

os.makedirs("data", exist_ok=True)

dfs = []
for split in ["train", "test", "dev"]:
    path = f"data/{split}.tsv"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    print(f"{split}: {len(df)} rows | columns: {df.columns.tolist()}")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

label_map = {"false": 1, "mixture": 1, "true": 0, "unproven": 1}
df = df[["claim", "label"]].dropna()
df = df[df["label"].isin(label_map)]
df["label"] = df["label"].map(label_map)
df = df.rename(columns={"claim": "text"})

df.to_csv("data/health_fact.csv", index=False)
print(f"\nSaved {len(df)} samples to data/health_fact.csv")
print(df["label"].value_counts())