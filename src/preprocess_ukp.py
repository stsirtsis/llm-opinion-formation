import glob
import os

import pandas as pd

INPUT_DIR = os.path.join("data", "original", "UKP_sentential_argument_mining", "data")
OUTPUT_PATH = os.path.join("data", "processed", "ukp.tsv")

# Read and concatenate all topic TSV files
input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tsv")))
dfs = [pd.read_csv(f, sep="\t", dtype=str, quoting=3, on_bad_lines="warn") for f in input_files]
df = pd.concat(dfs, ignore_index=True)

# Filter out NoArgument rows and rows with missing values
df = df[df["annotation"] != "NoArgument"]
df = df.dropna(subset=["sentence", "topic", "annotation"])

# Replace spaces in "topic" with underscores for consistency
df["topic"] = df["topic"].str.replace(" ", "_")

# Select and rename columns to standardized schema
df = df[["sentence", "topic", "annotation", "set"]].rename(columns={"set": "split"})

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
