#!/usr/bin/env python3

import gc
import torch
import json
import numpy as np
import pandas as pd
import os
import click
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# Constants
LABEL_MAP = {"Argument_against": 0, "Argument_for": 1}
LABEL_NAMES = ["Argument_against", "Argument_for"]
INSTRUCTION = "Classify the stance of the following text as either supporting or opposing {topic}."

# Hardcoded list of embedding models
EMBEDDING_MODELS = [
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "Salesforce/SFR-Embedding-Mistral",
    "Qwen/Qwen3-Embedding-8B",
    "Octen/Octen-Embedding-8B",
    "Linq-AI-Research/Linq-Embed-Mistral"
]


def load_data(data_file):
    """Load preprocessed data from a TSV file and add label/topic_formatted columns."""
    df = pd.read_csv(data_file, sep="\t", dtype=str, quoting=3)

    df["label"] = df["annotation"].map(LABEL_MAP)
    df["topic_formatted"] = df["topic"].str.replace('_', ' ').str.title()

    logging.info(f"Loaded {len(df)} samples from {data_file}")
    logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    logging.info(f"Split distribution: {df['split'].value_counts().to_dict()}")

    return df


def compute_class_centroids(embeddings, topics_list, labels):
    """Compute normalized mean embedding per (topic, class). Returns centroids and info for LOO."""
    label_to_class = {1: "for", 0: "against"}
    # Accumulate sums and counts
    info = {}
    for i, (topic, label) in enumerate(zip(topics_list, labels)):
        cls = label_to_class[int(label)]
        if topic not in info:
            info[topic] = {
                "for": {"sum": torch.zeros(embeddings.shape[1]), "count": 0},
                "against": {"sum": torch.zeros(embeddings.shape[1]), "count": 0},
            }
        info[topic][cls]["sum"] += embeddings[i]
        info[topic][cls]["count"] += 1

    centroids = {}
    for topic, cls_info in info.items():
        centroids[topic] = {}
        for cls in ("for", "against"):
            count = cls_info[cls]["count"]
            if count == 0:
                logging.warning(f"No samples for topic={topic}, class={cls}. Using zero vector.")
                centroids[topic][cls] = torch.zeros(embeddings.shape[1])
            else:
                mean = cls_info[cls]["sum"] / count
                norm = torch.norm(mean)
                centroids[topic][cls] = mean / norm if norm > 0 else mean

    return centroids, info


def predict_with_centroids(embeddings, topics_list, centroids, loo=False, centroid_info=None, labels=None):
    """Predict labels using nearest centroid, with optional leave-one-out correction."""
    label_to_class = {1: "for", 0: "against"}
    predictions = []
    sims_for = []
    sims_against = []

    for i, topic in enumerate(topics_list):
        emb = embeddings[i]

        if loo and labels is not None and centroid_info is not None:
            cls = label_to_class[int(labels[i])]
            # Compute LOO centroid for the sample's own class
            own_sum = centroid_info[topic][cls]["sum"]
            own_count = centroid_info[topic][cls]["count"]
            if own_count <= 1:
                logging.warning(f"LOO: only {own_count} sample(s) for topic={topic}, class={cls}. Using zero vector.")
                loo_centroid = torch.zeros_like(emb)
            else:
                loo_mean = (own_sum - emb) / (own_count - 1)
                loo_norm = torch.norm(loo_mean)
                loo_centroid = loo_mean / loo_norm if loo_norm > 0 else loo_mean

            # The other class centroid stays unchanged
            other_cls = "against" if cls == "for" else "for"
            other_centroid = centroids[topic][other_cls]

            if cls == "for":
                for_centroid = loo_centroid
                against_centroid = other_centroid
            else:
                for_centroid = other_centroid
                against_centroid = loo_centroid
        else:
            for_centroid = centroids[topic]["for"]
            against_centroid = centroids[topic]["against"]

        sim_for = torch.dot(emb, for_centroid).item()
        sim_against = torch.dot(emb, against_centroid).item()

        predictions.append(1 if sim_for > sim_against else 0)
        sims_for.append(sim_for)
        sims_against.append(sim_against)

    return np.array(predictions), np.array(sims_for), np.array(sims_against)


def sample_calibration_set(topics_list, labels, cal_size_per_class=20, seed=42):
    """Sample a balanced calibration subset (per topic per class) for temperature/weight tuning."""
    rng = np.random.RandomState(seed)
    cal_mask = np.zeros(len(labels), dtype=bool)

    # Group indices by (topic, label)
    groups = {}
    for i, (topic, label) in enumerate(zip(topics_list, labels)):
        key = (topic, int(label))
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    for key, indices in groups.items():
        n = min(cal_size_per_class, len(indices))
        selected = rng.choice(indices, size=n, replace=False)
        cal_mask[selected] = True

    return cal_mask


def encode_texts(embedding_model_obj, texts, topics_list, batch_size):
    """Batch encode texts with progress bar."""
    batch_embeddings_list = []
    formatted_texts = []
    for t, topic in zip(texts, topics_list):
        topic_formatted = topic.replace('_', ' ').title()
        instruction_formatted = INSTRUCTION.format(topic=topic_formatted)
        formatted_texts.append(f"Instruct: {instruction_formatted}\nQuery: {t}")

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", file=sys.stdout, mininterval=10.0):
        batch_texts = formatted_texts[i:i+batch_size]
        batch_embeddings = embedding_model_obj.encode(
            batch_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            prompt=None,
            normalize_embeddings=True
        )
        batch_embeddings_list.append(batch_embeddings.cpu().float())
        torch.cuda.empty_cache()

    return torch.cat(batch_embeddings_list, dim=0)


def calibrate_temperature(logits_for, logits_against, labels):
    """Find optimal temperature to minimize NLL on calibration set."""
    def nll_loss(T):
        # Stack logits: [against, for] per sample
        logits = np.stack([logits_against, logits_for], axis=1) / T
        # Log-softmax for numerical stability
        max_logits = logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(logits - max_logits).sum(axis=1, keepdims=True)) + max_logits
        log_probs = logits - log_sum_exp
        # NLL: -log(p(true_label))
        nll = -log_probs[np.arange(len(labels)), labels].mean()
        return nll

    result = minimize_scalar(nll_loss, bounds=(0.01, 10.0), method='bounded')
    return result.x



@click.command()
@click.option('--exp_name', type=str, required=True, help='Experiment name')
@click.option('--output_dir', type=str, required=True, help='Output directory for results')
@click.option('--output_filename', type=str, required=True, help='Output filename (auto-generated by executor)')
@click.option('--dataset', type=str, required=True, help='Dataset name (e.g., ukp, semeval)')
@click.option('--model_dir', type=str, required=True, help='Directory for caching embeddings')
@click.option('--embed_batch_size', type=int, default=32, help='Batch size for generating embeddings')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--cal_size', type=int, default=20, help='Calibration samples per class per topic')
def main(exp_name, output_dir, output_filename, dataset, model_dir, embed_batch_size, seed, cal_size):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    logging.info(f"Experiment: {exp_name}")
    logging.info(f"Model dir: {model_dir}")
    logging.info(f"Number of embedding models: {len(EMBEDDING_MODELS)}")
    logging.info("=" * 70)

    # Load all data
    data_file = os.path.join("data", "processed", f"{dataset}.tsv")
    all_df = load_data(data_file)
    topics = sorted(all_df['topic'].unique().tolist())

    labels = all_df["label"].values
    topics_list = all_df["topic"].tolist()

    # Sample calibration subset for temperature/weight tuning
    cal_mask = sample_calibration_set(topics_list, labels, cal_size_per_class=cal_size, seed=seed)
    cal_topics = [t for t, m in zip(topics_list, cal_mask) if m]
    cal_labels = labels[cal_mask]

    logging.info(f"Total samples: {len(labels)}")
    logging.info(f"Calibration samples: {cal_mask.sum()}")

    # Create embeddings cache directory
    embeddings_cache_dir = os.path.join(model_dir, f"embeddings_cache_{dataset}")
    os.makedirs(embeddings_cache_dir, exist_ok=True)

    # Process each embedding model
    model_results = {}

    for model_idx, embedding_model in enumerate(EMBEDDING_MODELS):
        logging.info(f"\n{'='*70}")
        logging.info(f"[{model_idx+1}/{len(EMBEDDING_MODELS)}] Processing: {embedding_model}")
        logging.info(f"{'='*70}")

        # Create cache filename
        model_name_sanitized = embedding_model.replace("/", "_").replace("\\", "_")
        cache_file = os.path.join(embeddings_cache_dir, f"{model_name_sanitized}_embeddings.pt")

        # Try to load cached embeddings
        if os.path.exists(cache_file):
            logging.info(f"Loading cached embeddings from {cache_file}...")
            cached_data = torch.load(cache_file, map_location='cpu', weights_only=False)
            all_embeddings = cached_data['all_embeddings']
            logging.info(f"Cached embeddings loaded: {all_embeddings.shape}")
        else:
            # Compute embeddings
            logging.info(f"Computing embeddings with {embedding_model}...")
            embedding_model_obj = SentenceTransformer(
                embedding_model,
                device=str(device),
                trust_remote_code=True,
                model_kwargs={"dtype": torch.bfloat16}
            )

            logging.info("Computing embeddings for all data...")
            all_embeddings = encode_texts(
                embedding_model_obj,
                all_df["sentence"].tolist(),
                all_df["topic"].tolist(),
                embed_batch_size
            )

            # Save to cache
            logging.info(f"Saving embeddings to cache: {cache_file}")
            torch.save({
                'all_embeddings': all_embeddings,
                'embedding_model_name': embedding_model
            }, cache_file)

            del embedding_model_obj
            gc.collect()
            torch.cuda.empty_cache()

        # Compute class centroids from ALL data
        centroids, centroid_info = compute_class_centroids(all_embeddings, topics_list, labels)

        # Save centroids for use in predict_opinion
        centroids_dir = os.path.join(model_dir, f"centroids_{dataset}")
        os.makedirs(centroids_dir, exist_ok=True)
        centroids_file = os.path.join(centroids_dir, f"{model_name_sanitized}_centroids.pt")
        torch.save(centroids, centroids_file)
        logging.info(f"Saved centroids to {centroids_file}")

        # Save centroid_info (sums and counts) for LOO in predict_opinion
        centroid_info_file = os.path.join(centroids_dir, f"{model_name_sanitized}_centroid_info.pt")
        torch.save(centroid_info, centroid_info_file)
        logging.info(f"Saved centroid_info to {centroid_info_file}")

        # Calibration predictions with LOO
        cal_embeddings = all_embeddings[cal_mask]
        cal_preds, cal_sims_for, cal_sims_against = predict_with_centroids(
            cal_embeddings, cal_topics, centroids,
            loo=True, centroid_info=centroid_info, labels=cal_labels
        )
        weight = accuracy_score(cal_labels, cal_preds)

        # Calibrate temperature on calibration set with LOO
        temperature = calibrate_temperature(cal_sims_for, cal_sims_against, cal_labels)

        logging.info(f"Calibration Accuracy (weight): {weight*100:.2f}%")
        logging.info(f"Calibrated temperature: {temperature:.4f}")

        # Full predictions on ALL data with LOO
        all_preds, all_sims_for, all_sims_against = predict_with_centroids(
            all_embeddings, topics_list, centroids,
            loo=True, centroid_info=centroid_info, labels=labels
        )
        # Calibrated P(for) via temperature-scaled softmax
        logits = np.stack([all_sims_against, all_sims_for], axis=1) / temperature
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        all_probs_for = (exp_logits / exp_logits.sum(axis=1, keepdims=True))[:, 1]

        all_acc = accuracy_score(labels, all_preds)
        all_f1 = f1_score(labels, all_preds, average="macro", zero_division=0)

        logging.info(f"All data: Accuracy={all_acc*100:.2f}%, F1={all_f1:.4f}")

        model_results[embedding_model] = {
            'weight': weight,
            'temperature': temperature,
            'all_probs_for': all_probs_for,
            'all_preds': all_preds,
            'all_acc': all_acc,
            'all_f1': all_f1
        }

    # Ensemble: weighted average of calibrated P(for), predict "for" if > 0.5
    logging.info(f"\n{'='*70}")
    logging.info("Creating ensemble with weighted probability averaging")
    logging.info(f"{'='*70}")

    weights = [r['weight'] for r in model_results.values()]
    all_probs = [r['all_probs_for'] for r in model_results.values()]

    logging.info("Model weights (calibration accuracies):")
    for model_name, result in model_results.items():
        logging.info(f"  {model_name}: {result['weight']:.4f}")

    weights_arr = np.array(weights)
    ensemble_prob = np.dot(weights_arr, np.array(all_probs)) / weights_arr.sum()
    ensemble_preds = (ensemble_prob > 0.5).astype(int)

    # Compute ensemble metrics
    ensemble_acc = accuracy_score(labels, ensemble_preds)
    ensemble_f1 = f1_score(labels, ensemble_preds, average="macro", zero_division=0)

    logging.info("\nEnsemble Performance (all data):")
    logging.info(f"  Accuracy: {ensemble_acc*100:.2f}%")
    logging.info(f"  Macro F1: {ensemble_f1:.4f}")

    # Confidence: P(predicted class) = max(ensemble_prob, 1 - ensemble_prob), in [0.5, 1]
    confidence = np.maximum(ensemble_prob, 1 - ensemble_prob)
    avg_confidence = np.mean(confidence)
    std_confidence = np.std(confidence)
    p5_confidence = np.percentile(confidence, 5)
    p95_confidence = np.percentile(confidence, 95)

    logging.info("\nConfidence (P(predicted class)):")
    logging.info(f"  Mean: {avg_confidence:.4f} +/- {std_confidence:.4f}")
    logging.info(f"  5th/95th percentiles: {p5_confidence:.4f} / {p95_confidence:.4f}")

    # Classification report
    report = classification_report(labels, ensemble_preds, target_names=LABEL_NAMES, zero_division=0)
    print(f"\nEnsemble Classification Report (all data):\n{report}")

    # Save per-sample ensemble predictions for downstream filtering
    ensemble_pred_labels = np.where(ensemble_preds == 1, 'Argument_for', 'Argument_against')
    correctly_classified = (all_df['annotation'].values == ensemble_pred_labels)
    predictions_df = pd.DataFrame({
        'sentence_id': all_df['sentence_id'].values,
        'correctly_classified': correctly_classified,
    })
    predictions_path = os.path.join(model_dir, f"ensemble_predictions_{dataset}.tsv")
    predictions_df.to_csv(predictions_path, sep="\t", index=False, quoting=3)
    n_correct = correctly_classified.sum()
    logging.info(f"Saved per-sample predictions to {predictions_path} ({n_correct}/{len(predictions_df)} correct)")

    # Save weights to JSON
    weights_json = {
        "models": {},
        "metadata": {
            "n_models": len(EMBEDDING_MODELS),
            "n_total": len(labels),
            "n_cal": int(cal_mask.sum()),
            "cal_size_per_class": cal_size,
            "topics": topics,
            "instruction": INSTRUCTION
        }
    }

    for model_name, result in model_results.items():
        short_name = model_name.split('/')[-1].lower()
        weights_json["models"][short_name] = {
            "full_name": model_name,
            "weight": float(result['weight']),
            "temperature": float(result['temperature'])
        }

    os.makedirs(model_dir, exist_ok=True)
    weights_json_path = os.path.join(model_dir, f"ensemble_weights_{dataset}.json")
    with open(weights_json_path, 'w') as f:
        json.dump(weights_json, f, indent=4)
    logging.info(f"\nSaved weights to {weights_json_path}")

    # Save results text file
    results_path = os.path.join(model_dir, f"ensemble_results_{dataset}.txt")
    with open(results_path, 'w') as f:
        f.write("Ensemble Classifier Results (Centroid-based, Weighted Probability Averaging)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of models: {len(EMBEDDING_MODELS)}\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Calibration samples: {cal_mask.sum()} ({cal_size} per class per topic)\n\n")
        f.write("Classification method: Nearest centroid (mean embedding per topic per class)\n")
        f.write("Calibration: LOO on calibration subset for temperature/weight tuning\n")
        f.write("Ensemble method: Weighted probability averaging\n")
        f.write("  - Each model produces calibrated P(for) via temperature-scaled softmax\n")
        f.write("  - Ensemble P(for) = weighted average of per-model P(for)\n")
        f.write("  - Final prediction: for if ensemble P(for) > 0.5, else against\n\n")
        f.write("Model Weights (Calibration Accuracies):\n")
        for model_name, result in model_results.items():
            f.write(f"  {model_name}: {result['weight']:.4f}\n")
        f.write("\nIndividual Model Performances (all data):\n")
        for model_name, result in model_results.items():
            f.write(f"  {model_name}: Accuracy={result['all_acc']*100:.2f}%, F1={result['all_f1']:.4f}\n")
        f.write("\nEnsemble Performance (all data):\n")
        f.write(f"  Accuracy: {ensemble_acc*100:.2f}%\n")
        f.write(f"  Macro F1: {ensemble_f1:.4f}\n\n")
        f.write("Confidence (P(predicted class), in [0.5, 1]):\n")
        f.write(f"  Mean: {avg_confidence:.4f} +/- {std_confidence:.4f}\n")
        f.write(f"  5th/95th percentiles: {p5_confidence:.4f} / {p95_confidence:.4f}\n\n")
        f.write("Classification Report (all data):\n")
        f.write(report)
        f.write("\nPer-Topic Performance:\n")
        topics_arr = np.array(topics_list)
        for topic in topics:
            mask = topics_arr == topic
            n = mask.sum()
            f.write(f"\n  {topic} (N={n}):\n")
            for model_name, result in model_results.items():
                short = model_name.split('/')[-1]
                t_acc = accuracy_score(labels[mask], result['all_preds'][mask])
                t_f1 = f1_score(labels[mask], result['all_preds'][mask], average="macro", zero_division=0)
                f.write(f"    {short:<35s} Acc={t_acc*100:.2f}%, F1={t_f1:.4f}\n")
            ens_acc = accuracy_score(labels[mask], ensemble_preds[mask])
            ens_f1 = f1_score(labels[mask], ensemble_preds[mask], average="macro", zero_division=0)
            f.write(f"    {'Ensemble':<35s} Acc={ens_acc*100:.2f}%, F1={ens_f1:.4f}\n")

    logging.info(f"Saved results to {results_path}")

    # Print final summary
    logging.info("\n" + "="*70)
    logging.info("FINAL ENSEMBLE RESULTS (all data)")
    logging.info("="*70)
    logging.info(f"Number of models: {len(EMBEDDING_MODELS)}")
    logging.info("Ensemble method: Centroid-based, weighted probability averaging")
    logging.info(f"Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    logging.info(f"Ensemble F1: {ensemble_f1:.4f}")


if __name__ == "__main__":
    main()
