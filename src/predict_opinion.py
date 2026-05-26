#!/usr/bin/env python3

import gc
import torch
import json
import numpy as np
import pandas as pd
import os
import random
import click
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
import sys

from utils import task_dataset_compatible, TASK_ALLOWED_DATASETS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ensemble_weights(weights_file):
    """Load ensemble model weights and metadata from JSON file."""
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)

    models = weights_data.get('models', {})
    metadata = weights_data.get('metadata', {})

    logging.info(f"Loaded ensemble weights from {weights_file}")
    logging.info(f"Number of models: {len(models)}")

    return models, metadata


def load_centroids(centroids_dir, model_name, topic):
    """Load precomputed class centroids for a given model and topic."""
    model_name_sanitized = model_name.replace("/", "_").replace("\\", "_")
    centroids_file = os.path.join(centroids_dir, f"{model_name_sanitized}_centroids.pt")
    centroids = torch.load(centroids_file, map_location='cpu', weights_only=False)
    if topic not in centroids:
        raise ValueError(f"Topic '{topic}' not found in centroids. Available: {list(centroids.keys())}")
    logging.info(f"Loaded centroids for topic '{topic}' from {centroids_file}")
    return centroids[topic]


def load_centroid_info(centroids_dir, model_name, topic):
    """Load centroid accumulator info (sums and counts) for LOO prediction. Returns None if unavailable."""
    model_name_sanitized = model_name.replace("/", "_").replace("\\", "_")
    info_file = os.path.join(centroids_dir, f"{model_name_sanitized}_centroid_info.pt")
    if not os.path.exists(info_file):
        logging.warning(f"Centroid info file not found: {info_file}. LOO will not be applied.")
        return None
    info = torch.load(info_file, map_location='cpu', weights_only=False)
    if topic not in info:
        logging.warning(f"Topic '{topic}' not found in centroid_info. LOO will not be applied.")
        return None
    logging.info(f"Loaded centroid_info for topic '{topic}' from {info_file}")
    return info[topic]


def encode_texts_cached(embedding_model_obj, texts, topic, instruction, batch_size, model_name, cache_dir, cache_key):
    """Encode texts with instruction prefix, loading from cache if available."""
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache filename
    model_name_sanitized = model_name.replace("/", "_").replace("\\", "_")
    cache_file = os.path.join(cache_dir, f"{model_name_sanitized}_{cache_key}_embeddings.pt")

    # Try to load cached embeddings
    if os.path.exists(cache_file):
        logging.info(f"Loading cached embeddings from {cache_file}...")
        embeddings = torch.load(cache_file, map_location='cpu', weights_only=False)
        logging.info(f"Cached embeddings loaded: {embeddings.shape}")
        return embeddings

    # Compute embeddings
    logging.info("Computing embeddings (no cache found)...")
    batch_embeddings_list = []
    topic_formatted = topic.replace('_', ' ').title()
    instruction_formatted = instruction.format(topic=topic_formatted)
    formatted_texts = [f"Instruct: {instruction_formatted}\nQuery: {t}" for t in texts]

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

    embeddings = torch.cat(batch_embeddings_list, dim=0)

    # Save to cache
    logging.info(f"Saving embeddings to cache: {cache_file}")
    torch.save(embeddings, cache_file)

    return embeddings


def load_from_ensemble_cache(sentence_ids, embedding_model_name, cache_dir):
    """Load precomputed embeddings from build_ensemble's cache by sentence_id. Returns None if unavailable."""
    model_name_sanitized = embedding_model_name.replace("/", "_").replace("\\", "_")
    cache_file = os.path.join(cache_dir, f"{model_name_sanitized}_embeddings.pt")

    if not os.path.exists(cache_file):
        return None

    try:
        cached_data = torch.load(cache_file, map_location='cpu', weights_only=False)
        all_embeddings = cached_data['all_embeddings']

        # sentence_id is 1-indexed, convert to 0-indexed for array access
        indices = [int(sid) - 1 for sid in sentence_ids]
        embeddings = all_embeddings[indices]
        logging.info(f"Loaded {len(sentence_ids)} sentence embeddings from ensemble cache: {cache_file}")
        return embeddings

    except Exception as e:
        logging.info(f"Failed to load from ensemble cache: {e}")
        return None


def predict_single_model(embeddings, reference_embeddings, temperature,
                         loo_centroid_info=None, loo_embeddings=None, loo_labels=None):
    """Predict labels and calibrated P(for) using temperature-scaled softmax, with optional LOO correction."""
    for_emb = reference_embeddings["for"]
    against_emb = reference_embeddings["against"]

    predictions = []
    prob_for_list = []

    for i in range(len(embeddings)):
        emb = embeddings[i]

        if loo_centroid_info is not None and loo_embeddings is not None and loo_labels is not None:
            own_cls = loo_labels[i]

            # LOO: remove this point's embedding from its own class centroid
            own_sum = loo_centroid_info[own_cls]["sum"]
            own_count = loo_centroid_info[own_cls]["count"]
            if own_count <= 1:
                loo_centroid = torch.zeros_like(emb)
            else:
                loo_mean = (own_sum - loo_embeddings[i]) / (own_count - 1)
                loo_norm = torch.norm(loo_mean)
                loo_centroid = loo_mean / loo_norm if loo_norm > 0 else loo_mean

            other_cls = "against" if own_cls == "for" else "for"
            if own_cls == "for":
                cur_for_emb, cur_against_emb = loo_centroid, reference_embeddings[other_cls]
            else:
                cur_for_emb, cur_against_emb = reference_embeddings[other_cls], loo_centroid
        else:
            cur_for_emb = for_emb
            cur_against_emb = against_emb

        # Compute cosine similarities (dot product since normalized)
        sim_for = torch.dot(emb, cur_for_emb).item()
        sim_against = torch.dot(emb, cur_against_emb).item()

        # Temperature-scaled softmax
        logits = np.array([sim_against, sim_for]) / temperature
        # Numerical stability: subtract max before exp
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        prob_for = probs[1]

        # Prediction based on probability
        predictions.append(1 if prob_for > 0.5 else 0)
        prob_for_list.append(prob_for)

    return np.array(predictions), np.array(prob_for_list)


def ensemble_predict_with_confidence(all_probs_for, weights):
    """Compute ensemble predictions via weighted average of calibrated probabilities."""
    weights = np.array(weights)
    all_probs = np.array(all_probs_for)  # Shape: (n_models, n_samples)

    # Weighted average of probabilities
    ensemble_prob = np.dot(weights, all_probs) / weights.sum()

    # Final prediction: P(for) > 0.5 -> for (1), otherwise -> against (0)
    predictions = (ensemble_prob > 0.5).astype(int)

    return predictions, ensemble_prob


def predict_column(texts, topic, ensemble_models, instruction, centroids_dir,
                   embed_batch_size, embeddings_cache_dir, cache_key, device,
                   ensemble_cache_dir=None, sentence_ids=None,
                   quantification_method="centroid",
                   loo=False, loo_labels=None):
    """Run ensemble prediction on a list of texts. Returns prediction strings and P(for) confidences."""
    # Process each embedding model
    all_predictions = []
    all_probs_for = []
    weights_list = []

    for model_idx, (short_name, model_info) in enumerate(ensemble_models.items()):
        full_name = model_info['full_name']
        weight = model_info['weight']
        temperature = model_info['temperature']

        logging.info(f"\n{'='*70}")
        logging.info(f"[{model_idx+1}/{len(ensemble_models)}] Processing: {full_name}")
        logging.info(f"Weight: {weight:.4f}, Temperature: {temperature:.4f}")
        logging.info(f"Quantification method: {quantification_method}")
        logging.info(f"{'='*70}")

        if quantification_method == "centroid":
            # Load precomputed centroids
            reference_embeddings = load_centroids(centroids_dir, full_name, topic)

            # Load centroid_info for LOO if requested
            loo_centroid_info = None
            loo_embeddings = None
            if loo:
                loo_centroid_info = load_centroid_info(centroids_dir, full_name, topic)

            # Try to load embeddings from build_ensemble's cache
            embeddings = None
            if ensemble_cache_dir is not None and sentence_ids is not None:
                embeddings = load_from_ensemble_cache(
                    sentence_ids, full_name, ensemble_cache_dir
                )
                if embeddings is not None and loo_centroid_info is not None:
                    # These embeddings are the same ones used to build the centroids
                    loo_embeddings = embeddings

            if embeddings is None:
                # Load embedding model
                logging.info("Loading embedding model...")
                embedding_model_obj = SentenceTransformer(
                    full_name,
                    device=str(device),
                    trust_remote_code=True,
                    model_kwargs={"dtype": torch.bfloat16}
                )

                # Encode input texts (with caching)
                embeddings = encode_texts_cached(
                    embedding_model_obj, texts, topic, instruction, embed_batch_size,
                    full_name, embeddings_cache_dir, cache_key
                )

                if loo_centroid_info is not None:
                    # Freshly encoded embeddings match the ones used for centroids
                    loo_embeddings = embeddings

                del embedding_model_obj
                gc.collect()
                torch.cuda.empty_cache()

        else:
            raise ValueError(f"Unknown quantification_method: {quantification_method}")

        # Predict via temperature-scaled softmax
        predictions, prob_for = predict_single_model(
            embeddings, reference_embeddings, temperature,
            loo_centroid_info=loo_centroid_info, loo_embeddings=loo_embeddings,
            loo_labels=loo_labels
        )

        all_predictions.append(predictions)
        all_probs_for.append(prob_for)
        weights_list.append(weight)

        # Log individual model stats
        n_for = (predictions == 1).sum()
        n_against = (predictions == 0).sum()
        mean_prob = prob_for.mean()
        logging.info(f"Predictions: {n_for} for, {n_against} against, mean P(for): {mean_prob:.4f}")

        # Clean up embeddings
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()

    # Ensemble predictions with calibrated probabilities
    logging.info(f"\n{'='*70}")
    logging.info("Computing ensemble predictions with weighted probability averaging")
    logging.info(f"{'='*70}")

    ensemble_preds, confidence = ensemble_predict_with_confidence(all_probs_for, weights_list)

    # Convert to string labels
    predictions_str = np.where(ensemble_preds == 1, 'for', 'against')

    # Log summary statistics
    n_for = (ensemble_preds == 1).sum()
    n_against = (ensemble_preds == 0).sum()
    logging.info(f"Ensemble predictions: {n_for} for, {n_against} against")
    logging.info(f"Ensemble P(for) statistics: mean={confidence.mean():.4f}, min={confidence.min():.4f}, max={confidence.max():.4f}")

    return predictions_str, confidence


@click.command()
@click.option('--exp_name', type=str, required=True, help='Experiment name')
@click.option('--output_dir', type=str, required=True, help='Output directory for results')
@click.option('--output_filename', type=str, required=True, help='Output filename (auto-generated by executor)')
@click.option('--dataset', type=str, required=True, help='Dataset name (e.g., ukp, semeval)')
@click.option('--task', type=str, required=True, help='Task name (e.g., writing, improvement) — used to locate the transform_text input file')
@click.option('--weights_dir', type=str, required=True, help='Directory containing ensemble weights JSON files')
@click.option('--model', type=str, required=True, help='Model name used to generate transformed texts')
@click.option('--topic', type=str, required=True, help='Topic for opinion classification')
@click.option('--processed_data_dir', type=str, required=True, help='Directory containing transformed text files')
@click.option('--instruction', type=str, required=True, help='Instruction prompt for embedding model')
@click.option('--embed_batch_size', type=int, default=16, help='Batch size for generating embeddings')
@click.option('--embeddings_cache_dir', type=str, default='models/embeddings_cache_predict', help='Directory for caching embeddings')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--quantification_method', type=str, default='centroid', help='Quantification method (currently only centroid is supported)')
@click.option('--input_prefix', type=str, default='transform_text', help='Input file prefix (transform_text or steer_prompt_transform)')
@click.option('--prompt_variant', type=str, default='', help='Prompt variant used in transform step (for input filename construction)')
def main(exp_name, output_dir, output_filename, dataset, task, weights_dir, model, topic, processed_data_dir,
         instruction, embed_batch_size, embeddings_cache_dir, seed,
         quantification_method, input_prefix, prompt_variant):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Skip combinations where the task is not designed for this dataset.
    if not task_dataset_compatible(task, dataset):
        allowed = sorted(TASK_ALLOWED_DATASETS[task])
        logging.info(f"Task '{task}' is not compatible with dataset '{dataset}' (expected one of {allowed}). Skipping.")
        return

    # Set seeds
    set_seeds(seed)

    # Construct input file path from dataset, task, model, topic, and processed_data_dir
    model_sanitized = model.replace("/", "_")
    variant_part = f"__prompt_variant={prompt_variant}" if prompt_variant else ""
    input_file = os.path.join(
        processed_data_dir,
        f"{input_prefix}__dataset={dataset}__task={task}__model={model_sanitized}__topic={topic}{variant_part}.tsv"
    )

    logging.info(f"Experiment: {exp_name}")
    logging.info(f"Model: {model}")
    logging.info(f"Topic: {topic}")
    logging.info(f"Quantification method: {quantification_method}")
    logging.info(f"Input file: {input_file}")
    logging.info("=" * 70)

    if not os.path.exists(input_file):
        logging.info(f"Input file not found: {input_file}. Skipping.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load ensemble weights
    weights_file = os.path.join(weights_dir, f"ensemble_weights_{dataset}.json")
    ensemble_models, _ = load_ensemble_weights(weights_file)

    if len(ensemble_models) == 0:
        raise ValueError(f"No models found in weights file: {weights_file}")

    # Derive centroids directory
    centroids_dir = os.path.join(weights_dir, f"centroids_{dataset}")
    logging.info(f"Centroids directory: {centroids_dir}")

    # Per-dataset cache dir for predict embeddings
    embeddings_cache_dir = f"{embeddings_cache_dir}_{dataset}"

    # Ensemble cache for reusing build_ensemble's precomputed sentence embeddings
    ensemble_cache_dir = os.path.join(weights_dir, f"embeddings_cache_{dataset}")

    # Load input file
    logging.info(f"Loading input file: {input_file}")
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, dtype=str)
    else:
        df = pd.read_csv(input_file, sep="\t", dtype=str, quoting=3, on_bad_lines='warn')

    # Verify required columns exist
    required_columns = ['sentence', 'transformed_text', 'sentence_id', 'prompt_id', 'output_id', 'system_prompt_id']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input file. Available columns: {list(df.columns)}")

    # Drop rows with missing text in either column
    original_len = len(df)
    df = df.dropna(subset=['sentence', 'transformed_text']).reset_index(drop=True)

    if len(df) < original_len:
        logging.info(f"Dropped {original_len - len(df)} rows with missing text")

    logging.info(f"Total samples to classify: {len(df)}")

    # Predict for sentence column (deduplicated by sentence_id)
    logging.info("\n" + "=" * 70)
    logging.info("PREDICTING FOR 'sentence' COLUMN (deduplicated)")
    logging.info("=" * 70)

    # Get unique sentences by sentence_id to avoid redundant classification
    unique_cols = ['sentence_id', 'sentence']
    if 'annotation' in df.columns:
        unique_cols.append('annotation')
    unique_originals = df.drop_duplicates(subset=['sentence_id'])[unique_cols]
    logging.info(f"Unique original texts: {len(unique_originals)} (out of {len(df)} total rows)")

    # Map annotation labels to "for"/"against" for LOO centroid correction
    annotation_to_cls = {"Argument_for": "for", "Argument_against": "against"}
    sentence_loo_labels = None
    if 'annotation' in unique_originals.columns:
        sentence_loo_labels = [annotation_to_cls.get(a) for a in unique_originals['annotation']]
        if any(label is None for label in sentence_loo_labels):
            logging.warning("Some annotations could not be mapped for LOO. Disabling LOO.")
            sentence_loo_labels = None

    cache_key_sentence = f"{model_sanitized}_{topic}_sentence".replace('.', '_')
    pred_unique, conf_unique = predict_column(
        unique_originals['sentence'].tolist(), topic, ensemble_models, instruction, centroids_dir,
        embed_batch_size, embeddings_cache_dir, cache_key_sentence, device,
        ensemble_cache_dir=ensemble_cache_dir,
        sentence_ids=unique_originals['sentence_id'].tolist(),
        quantification_method=quantification_method,
        loo=True, loo_labels=sentence_loo_labels
    )

    # Create mapping from sentence_id to predictions
    unique_originals = unique_originals.copy()
    unique_originals['pred'] = pred_unique
    unique_originals['conf'] = conf_unique
    sentence_id_to_pred = unique_originals.set_index('sentence_id')[['pred', 'conf']].to_dict('index')

    # Map back to all rows
    pred_sentence = np.array([sentence_id_to_pred[sid]['pred'] for sid in df['sentence_id']])
    conf_sentence = np.array([sentence_id_to_pred[sid]['conf'] for sid in df['sentence_id']])

    # Predict for transformed_text column
    logging.info("\n" + "=" * 70)
    logging.info("PREDICTING FOR 'transformed_text' COLUMN")
    logging.info("=" * 70)

    variant_suffix = f"_{prompt_variant}" if prompt_variant else ""
    cache_key_transformed = f"{model_sanitized}_{topic}_transformed_text{variant_suffix}".replace('.', '_')
    pred_transformed, conf_transformed = predict_column(
        df['transformed_text'].tolist(), topic, ensemble_models, instruction, centroids_dir,
        embed_batch_size, embeddings_cache_dir, cache_key_transformed, device,
        quantification_method=quantification_method,
    )

    # Build output dataframe with specified column order
    output_df = pd.DataFrame({
        'sentence_id': df['sentence_id'],
        'prompt_id': df['prompt_id'],
        'output_id': df['output_id'],
        'original': df['sentence'],
        'transformed': df['transformed_text'],
        'prediction_original': pred_sentence,
        'confidence_original': conf_sentence,
        'prediction_transformed': pred_transformed,
        'confidence_transformed': conf_transformed,
    })

    # Add other original columns (except the ones we already included)
    already_included = {'sentence_id', 'prompt_id', 'output_id', 'sentence', 'transformed_text'}  # Original df column names
    for col in df.columns:
        if col not in already_included:
            output_df[col] = df[col].values

    # Save output
    output_path = os.path.join(output_dir, output_filename.replace('.json', '.tsv'))
    output_df.to_csv(output_path, sep="\t", index=False, quoting=3)
    logging.info(f"\nSaved predictions to {output_path}")

    # Print final summary
    logging.info("\n" + "=" * 70)
    logging.info("PREDICTION COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Input: {input_file}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Total samples: {len(output_df)}")

    n_for_sent = (pred_sentence == 'for').sum()
    n_against_sent = (pred_sentence == 'against').sum()
    n_for_trans = (pred_transformed == 'for').sum()
    n_against_trans = (pred_transformed == 'against').sum()

    logging.info(f"Sentence predictions: {n_for_sent} for ({100*n_for_sent/len(output_df):.1f}%), {n_against_sent} against ({100*n_against_sent/len(output_df):.1f}%)")
    logging.info(f"Transformed predictions: {n_for_trans} for ({100*n_for_trans/len(output_df):.1f}%), {n_against_trans} against ({100*n_against_trans/len(output_df):.1f}%)")


if __name__ == "__main__":
    main()
