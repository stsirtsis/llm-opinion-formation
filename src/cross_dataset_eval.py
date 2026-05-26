#!/usr/bin/env python3

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from build_ensemble import (
    load_data,
    compute_class_centroids,
    predict_with_centroids,
    EMBEDDING_MODELS,
)

TOPIC = "abortion"
MODEL_DIR = "models"
DATA_DIR = os.path.join("data", "processed")
DATASETS = ["ukp", "semeval"]


def load_abortion_data(dataset):
    """Load dataset and return abortion-only subset with indices into the full TSV."""
    data_file = os.path.join(DATA_DIR, f"{dataset}.tsv")
    df = load_data(data_file)
    mask = df["topic"] == TOPIC
    indices = np.where(mask.values)[0]
    df_abortion = df.loc[mask].reset_index(drop=True)
    return df_abortion, indices


def load_cached_embeddings(dataset, model_name, indices):
    """Load cached embeddings and slice to abortion rows."""
    model_name_sanitized = model_name.replace("/", "_").replace("\\", "_")
    cache_file = os.path.join(MODEL_DIR, f"embeddings_cache_{dataset}", f"{model_name_sanitized}_embeddings.pt")
    cached = torch.load(cache_file, map_location="cpu", weights_only=False)
    all_emb = cached["all_embeddings"]
    return all_emb[indices]


def load_centroids(dataset, model_name):
    """Load precomputed centroids and return only the abortion entry."""
    model_name_sanitized = model_name.replace("/", "_").replace("\\", "_")
    centroids_file = os.path.join(MODEL_DIR, f"centroids_{dataset}", f"{model_name_sanitized}_centroids.pt")
    centroids = torch.load(centroids_file, map_location="cpu", weights_only=False)
    return {TOPIC: centroids[TOPIC]}


def load_ensemble_weights(dataset):
    """Load ensemble weights and temperatures for a dataset, aligned with EMBEDDING_MODELS."""
    path = os.path.join(MODEL_DIR, f"ensemble_weights_{dataset}.json")
    with open(path) as f:
        data = json.load(f)
    weights, temperatures = [], []
    for model_name in EMBEDDING_MODELS:
        short = model_name.split("/")[-1].lower()
        weights.append(data["models"][short]["weight"])
        temperatures.append(data["models"][short]["temperature"])
    return weights, temperatures


def evaluate_direction(embeddings, topics_list, labels, centroids, temperature, loo, centroid_info=None):
    """Run prediction and return accuracy, macro-F1, and calibrated P(for)."""
    preds, sims_for, sims_against = predict_with_centroids(
        embeddings, topics_list, centroids,
        loo=loo, centroid_info=centroid_info, labels=labels if loo else None,
    )
    logits = np.stack([sims_against, sims_for], axis=1) / temperature
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    prob_for = (exp_logits / exp_logits.sum(axis=1, keepdims=True))[:, 1]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return preds, acc, f1, prob_for


def main():
    output_path = os.path.join(MODEL_DIR, "cross_dataset_eval.txt")
    outfile = open(output_path, "w")

    def log(msg=""):
        print(msg)
        outfile.write(msg + "\n")

    # Load abortion subsets for both datasets
    data = {}
    for ds in DATASETS:
        df, indices = load_abortion_data(ds)
        data[ds] = {
            "df": df,
            "indices": indices,
            "labels": df["label"].values,
            "topics_list": df["topic"].tolist(),
        }
        log(f"{ds}: {len(df)} abortion samples  (for={sum(df['label']==1)}, against={sum(df['label']==0)})")

    # Load ensemble weights and temperatures (source dataset provides both)
    weights, temperatures = {}, {}
    for ds in DATASETS:
        weights[ds], temperatures[ds] = load_ensemble_weights(ds)

    # Directions: (label, source_dataset, target_dataset, use_loo)
    directions = [
        ("UKP→UKP",         "ukp",    "ukp",    True),
        ("SemEval→SemEval",  "semeval","semeval", True),
        ("UKP→SemEval",      "ukp",    "semeval", False),
        ("SemEval→UKP",      "semeval","ukp",     False),
    ]

    # Collect per-model results: {direction_label: {model: (acc, f1)}}
    results = {d[0]: {} for d in directions}
    # Collect per-model P(for): {direction_label: [prob_for_per_model]}
    all_probs = {d[0]: [] for d in directions}

    for model_idx, model_name in enumerate(EMBEDDING_MODELS):
        short = model_name.split("/")[-1]
        log(f"\n[{model_idx+1}/{len(EMBEDDING_MODELS)}] {short}")

        # Load embeddings and centroids for both datasets
        emb = {}
        centroids = {}
        centroid_info = {}
        for ds in DATASETS:
            emb[ds] = load_cached_embeddings(ds, model_name, data[ds]["indices"])
            centroids[ds] = load_centroids(ds, model_name)
            # Recompute centroid_info for LOO (need sum/count)
            _, cinfo = compute_class_centroids(
                emb[ds], data[ds]["topics_list"], data[ds]["labels"]
            )
            centroid_info[ds] = cinfo

        for label, src, tgt, loo in directions:
            preds, acc, f1, prob_for = evaluate_direction(
                emb[tgt], data[tgt]["topics_list"], data[tgt]["labels"],
                centroids[src], temperature=temperatures[src][model_idx], loo=loo,
                centroid_info=centroid_info[tgt] if loo else None,
            )
            results[label][short] = (acc, f1)
            all_probs[label].append(prob_for)
            log(f"  {label:20s}  Acc={acc*100:5.1f}%  F1={f1:.3f}")

    # Ensemble: weighted average of calibrated P(for), predict "for" if > 0.5
    log(f"\n{'='*70}")
    log("ENSEMBLE (weighted probability averaging)")
    log(f"{'='*70}")

    ensemble_results = {}
    for label, src, tgt, loo in directions:
        w = np.array(weights[src])
        ens_prob = np.dot(w, np.array(all_probs[label])) / w.sum()
        ens_preds = (ens_prob > 0.5).astype(int)
        acc = accuracy_score(data[tgt]["labels"], ens_preds)
        f1 = f1_score(data[tgt]["labels"], ens_preds, average="macro", zero_division=0)
        ensemble_results[label] = (acc, f1)
        log(f"  {label:20s}  Acc={acc*100:5.1f}%  F1={f1:.3f}")

    # Summary table
    log(f"\n{'='*70}")
    log("SUMMARY TABLE")
    log(f"{'='*70}")

    header = f"{'Model':<35s}"
    for label, _, _, _ in directions:
        header += f"  {label:>16s}"
    log(header)
    log("-" * len(header))

    for model_name in EMBEDDING_MODELS:
        short = model_name.split("/")[-1]
        row = f"{short:<35s}"
        for label, _, _, _ in directions:
            acc, f1 = results[label][short]
            row += f"  {acc*100:5.1f}% / {f1:.3f}"
        log(row)

    row = f"{'Ensemble':<35s}"
    for label, _, _, _ in directions:
        acc, f1 = ensemble_results[label]
        row += f"  {acc*100:5.1f}% / {f1:.3f}"
    log(row)

    outfile.close()
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
