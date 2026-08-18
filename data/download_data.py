import pandas as pd
import requests
import os

os.makedirs("data", exist_ok=True)

splits = {
    "train": "https://raw.githubusercontent.com/neemakot/Health-Fact-Checking/master/data/PUBHEALTH/train.tsv",
    "test":  "https://raw.githubusercontent.com/neemakot/Health-Fact-Checking/master/data/PUBHEALTH/test.tsv",
    "dev":   "https://raw.githubusercontent.com/neemakot/Health-Fact-Checking/master/data/PUBHEALTH/dev.tsv",
}

dfs = []
for split, url in splits.items():
    print(f"Downloading {split}...")
    r = requests.get(url)
    from io import StringIO
    df = pd.read_csv(StringIO(r.text), sep="\t", on_bad_lines="skip")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print("Columns found:", df.columns.tolist())

label_map = {"false": 1, "mixture": 1, "true": 0, "unproven": 1}
df = df[["claim", "label"]].dropna()
df = df[df["label"].isin(label_map)]
df["label"] = df["label"].map(label_map)
df = df.rename(columns={"claim": "text"})

df.to_csv("data/health_fact.csv", index=False)
print(f"\nSaved {len(df)} samples to data/health_fact.csv")
print(df["label"].value_counts())