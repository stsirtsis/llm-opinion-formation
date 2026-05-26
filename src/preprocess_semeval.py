import os

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = os.path.join("data", "original", "semeval-data-all-annotations")
OUTPUT_PATH = os.path.join("data", "processed", "semeval.tsv")

INPUT_FILES = [
    "trainingdata-all-annotations.txt",
    "testdata-taskA-all-annotations.txt",
    "testdata-taskB-all-annotations.txt",
    "trialdata-all-annotations.txt",
]

STANCE_MAP = {
    "FAVOR": "Argument_for",
    "AGAINST": "Argument_against",
}

TOPIC_MAP = {
    "Atheism": "atheism",
    "Climate Change is a Real Concern": "acknowledging_climate_change",
    "Feminist Movement": "feminism",
    "Hillary Clinton": "hillary_clinton",
    "Legalization of Abortion": "abortion",
    "Donald Trump": "donald_trump",
}

VAL_FRACTION = 0.08
SEED = 42

# Read and concatenate all annotation files
dfs = [
    pd.read_csv(os.path.join(INPUT_DIR, f), sep="\t", dtype=str, quoting=3, on_bad_lines="warn", encoding="latin-1")
    for f in INPUT_FILES
]
df = pd.concat(dfs, ignore_index=True)

# Drop NONE stance and rows with missing values
df = df[df["Stance"] != "NONE"]
df = df.dropna(subset=["Tweet", "Target", "Stance"])

# Map labels and topics to UKP convention
df["annotation"] = df["Stance"].map(STANCE_MAP)
df["topic"] = df["Target"].map(TOPIC_MAP)
df = df.rename(columns={"Tweet": "sentence"})

# Remove "#SemST" from sentences
df["sentence"] = df["sentence"].str.replace("#SemST", "", regex=False).str.strip()

# Create val split (~8% per topic, stratified by annotation)
df["split"] = "train"
for topic in df["topic"].unique():
    mask = df["topic"] == topic
    topic_idx = df.index[mask]
    _, val_idx = train_test_split(
        topic_idx,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=df.loc[topic_idx, "annotation"],
    )
    df.loc[val_idx, "split"] = "val"

# Select final columns
df = df[["sentence", "topic", "annotation", "split"]]

# Add sentence_id (1-indexed, globally unique)
df = df.reset_index(drop=True)
df.insert(0, "sentence_id", range(1, len(df) + 1))

# Save
df.to_csv(OUTPUT_PATH, sep="\t", index=False, quoting=3)

# Summary
print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
print(f"Topics: {sorted(df['topic'].unique())}")
print(f"Annotations: {df['annotation'].value_counts().to_dict()}")
print(f"Splits: {df['split'].value_counts().to_dict()}")
