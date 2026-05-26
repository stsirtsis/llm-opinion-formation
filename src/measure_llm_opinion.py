#!/usr/bin/env python3

import gc
import json
import logging
import os
import random
import sys

import click
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_prompts

# Try importing Mistral3-specific classes (requires transformers from main + mistral-common)
try:
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    MISTRAL3_AVAILABLE = True
except ImportError:
    MISTRAL3_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_mistral3_model(model_name):
    """Check if the model requires Mistral3-specific loading."""
    model_lower = model_name.lower()
    return "ministral-3" in model_lower or "mistral-3" in model_lower


def is_qwen_model(model_name):
    """Check if the model is a Qwen model that supports enable_thinking."""
    return "qwen" in model_name.lower()


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with appropriate classes based on model type."""
    if is_mistral3_model(model_name):
        if not MISTRAL3_AVAILABLE:
            raise ImportError(
                f"Model {model_name} requires Mistral3-specific classes. "
                "Please install transformers from main branch and mistral-common:\n"
                "  pip install git+https://github.com/huggingface/transformers\n"
                "  pip install mistral-common>=1.8.6"
            )
        logging.info("Loading Mistral3 model with specialized classes")
        tokenizer = MistralCommonBackend.from_pretrained(model_name)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto"
        )
    else:
        logging.info("Loading model with AutoModelForCausalLM")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    return model, tokenizer


def generate_response(model, tokenizer, model_name, system_prompt, user_prompt, max_new_tokens, temperature, top_p, response_prefix=''):
    """Generate a single response from the model given system/user prompts."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template with model-specific kwargs
    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if is_qwen_model(model_name):
        chat_template_kwargs["enable_thinking"] = False

    input_text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)

    # Append response prefix if provided (used to steer generation)
    if response_prefix:
        input_text = input_text + response_prefix

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
    )

    # Decode only the new tokens (excluding input)
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up quotation marks if present
    while ((response.startswith('"') and response.endswith('"')) or (response.startswith("'") and response.endswith("'"))):
        response = response[1:-1].strip()

    # Strip tabs and newlines to keep TSV output clean
    response = response.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

    return response


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


def predict_single_model(embeddings, reference_embeddings, temperature):
    """Predict labels and calibrated P(for) using temperature-scaled softmax."""
    for_emb = reference_embeddings["for"]
    against_emb = reference_embeddings["against"]

    predictions = []
    prob_for_list = []

    for i in range(len(embeddings)):
        emb = embeddings[i]

        # Compute cosine similarities (dot product since normalized)
        sim_for = torch.dot(emb, for_emb).item()
        sim_against = torch.dot(emb, against_emb).item()

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


def predict_texts(texts, topic, ensemble_models, instruction, centroids_dir,
                  embed_batch_size, embeddings_cache_dir, cache_key, device,
                  quantification_method="centroid"):
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

            # Try to load cached text embeddings first
            embeddings = None
            os.makedirs(embeddings_cache_dir, exist_ok=True)
            model_name_sanitized = full_name.replace("/", "_").replace("\\", "_")
            cache_file = os.path.join(embeddings_cache_dir, f"{model_name_sanitized}_{cache_key}_embeddings.pt")

            if os.path.exists(cache_file):
                logging.info(f"Loading cached embeddings from {cache_file}...")
                embeddings = torch.load(cache_file, map_location='cpu', weights_only=False)
                logging.info(f"Cached embeddings loaded: {embeddings.shape}")

            if embeddings is None:
                # Load embedding model only if cache miss
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

                del embedding_model_obj
                gc.collect()
                torch.cuda.empty_cache()

        else:
            raise ValueError(f"Unknown quantification_method: {quantification_method}")

        # Predict via temperature-scaled softmax
        predictions, prob_for = predict_single_model(embeddings, reference_embeddings, temperature)

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
@click.option('--output_filename', type=str, required=True, help='Output filename')
@click.option('--dataset', type=str, required=True, help='Dataset name (e.g., ukp, semeval)')
@click.option('--model', type=str, required=True, help='LLM to measure opinion of')
@click.option('--topic', type=str, required=True, help='Topic to generate statements about')
@click.option('--num_responses', type=int, default=10, help='Number of responses to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature')
@click.option('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
@click.option('--max_new_tokens', type=int, default=256, help='Maximum tokens to generate')
@click.option('--weights_dir', type=str, required=True, help='Directory containing ensemble weights JSON files')
@click.option('--instruction', type=str, required=True, help='Instruction prompt for embedding model')
@click.option('--embed_batch_size', type=int, default=16, help='Batch size for generating embeddings')
@click.option('--embeddings_cache_dir', type=str, default='models/embeddings_cache_opinion', help='Directory for caching embeddings')
@click.option('--prompts_file', type=str, default='configs/prompts.json', help='Path to centralized prompts JSON file')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--quantification_method', type=str, default='centroid', help='Quantification method (currently only centroid is supported)')
def main(exp_name, output_dir, output_filename, dataset, model, topic, num_responses, temperature, top_p,
         max_new_tokens, weights_dir, instruction,
         embed_batch_size, embeddings_cache_dir, prompts_file, seed,
         quantification_method):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set seeds
    set_seeds(seed)

    logging.info(f"Experiment: {exp_name}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Model: {model}")
    logging.info(f"Topic: {topic}")
    logging.info(f"Number of responses: {num_responses}")
    logging.info("=" * 70)

    # Construct dataset-specific paths
    weights_file = os.path.join(weights_dir, f"ensemble_weights_{dataset}.json")
    centroids_dir = os.path.join(weights_dir, f"centroids_{dataset}")
    logging.info(f"Weights file: {weights_file}")
    logging.info(f"Centroids directory: {centroids_dir}")

    # Validate that required files exist before expensive generation
    if not os.path.exists(weights_file):
        logging.info(f"Weights file not found: {weights_file}. Skipping.")
        return

    if quantification_method == "centroid":
        # Verify the topic exists in centroids for at least the first model
        ensemble_models_check, _ = load_ensemble_weights(weights_file)
        first_model_info = next(iter(ensemble_models_check.values()))
        first_full_name = first_model_info['full_name']
        model_name_sanitized_check = first_full_name.replace("/", "_").replace("\\", "_")
        centroids_file_check = os.path.join(centroids_dir, f"{model_name_sanitized_check}_centroids.pt")
        if not os.path.exists(centroids_file_check):
            logging.info(f"Centroids file not found: {centroids_file_check}. Skipping.")
            return
        centroids_check = torch.load(centroids_file_check, map_location='cpu', weights_only=False)
        if topic not in centroids_check:
            logging.info(f"Topic '{topic}' not found in centroids for dataset '{dataset}'. Skipping.")
            return
        del centroids_check

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Format topic for prompts
    topic_formatted = topic.replace('_', ' ').lower()

    # Sanitize model name for filenames
    model_sanitized = model.replace("/", "_").replace("\\", "_")

    # Paths (include dataset and quantification_method to avoid checkpoint collisions)
    generation_checkpoint_path = os.path.join(output_dir, f"generations_{dataset}_{model_sanitized}_{topic}_{quantification_method}.tsv")
    output_path = os.path.join(output_dir, output_filename.replace('.json', '.tsv'))

    # Load prompts from centralized file
    prompts_config = load_prompts(prompts_file, "measure_llm_opinion")
    generation_prompts = prompts_config['prompts']

    # Check if final output already exists with all rows
    total_expected = len(generation_prompts) * num_responses
    if os.path.exists(output_path):
        existing_output = pd.read_csv(output_path, sep="\t", dtype=str, quoting=3)
        if len(existing_output) >= total_expected:
            logging.info(f"Final output already exists with {len(existing_output)} rows. Nothing to do.")
            return

    # =====================================================================
    # PHASE 1: Generation
    # =====================================================================
    logging.info("\n" + "=" * 70)
    logging.info("PHASE 1: GENERATING STATEMENTS")
    logging.info("=" * 70)

    num_prompts = len(generation_prompts)
    total_responses = num_prompts * num_responses
    logging.info(f"Using {num_prompts} prompt(s), {num_responses} responses per prompt = {total_responses} total responses")

    # Check for existing generation checkpoint
    # completed_keys tracks (prompt_id, output_id) tuples of ints
    completed_keys = set()
    if os.path.exists(generation_checkpoint_path):
        existing_gen = pd.read_csv(generation_checkpoint_path, sep="\t", dtype=str, quoting=3)
        completed_keys = set(zip(existing_gen['prompt_id'].astype(int).tolist(), existing_gen['output_id'].astype(int).tolist()))
        logging.info(f"Resuming from generation checkpoint: {len(completed_keys)} responses already generated")

        if len(completed_keys) >= total_responses:
            logging.info("All responses already generated. Skipping generation phase.")
        else:
            logging.info(f"Remaining to generate: {total_responses - len(completed_keys)} responses")

    # Generate responses if needed
    if len(completed_keys) < total_responses:
        # Load LLM
        logging.info(f"Loading LLM: {model}")
        model_obj, tokenizer = load_model_and_tokenizer(model)
        logging.info("LLM loaded successfully!")

        system_prompt = prompts_config['system_prompt']

        # Generate responses for each prompt/prefix pair
        for prompt_id, prompt_entry in enumerate(generation_prompts, start=1):
            # Format prompts with topic
            user_prompt = prompt_entry['user_prompt'].format(topic=topic_formatted)
            response_prefix = prompt_entry.get('response_prefix', '').format(topic=topic_formatted)

            logging.info(f"\nPrompt {prompt_id}/{num_prompts}: \"{user_prompt}\"")
            logging.info(f"Prefix: \"{response_prefix}\"")

            for output_id in range(1, num_responses + 1):
                if (prompt_id, output_id) in completed_keys:
                    continue

                logging.info(f"  Generating response {output_id}/{num_responses} for prompt {prompt_id}...")

                response = generate_response(
                    model_obj, tokenizer, model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    response_prefix=response_prefix
                )

                # Prepend the response prefix to the generated text for the full statement
                full_response = response_prefix + response

                logging.info(f"    Generated: {full_response[:100]}...")

                # Checkpoint this response
                checkpoint_row = pd.DataFrame({
                    'prompt_id': [prompt_id],
                    'output_id': [output_id],
                    'topic': [topic],
                    'generated_text': [full_response]
                })
                header = not os.path.exists(generation_checkpoint_path)
                checkpoint_row.to_csv(generation_checkpoint_path, sep="\t", index=False, mode='a', header=header, quoting=3)

        # Unload LLM to free GPU memory
        logging.info("Unloading LLM to free GPU memory...")
        del model_obj
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 2: Opinion Prediction
    # =====================================================================
    logging.info("\n" + "=" * 70)
    logging.info("PHASE 2: PREDICTING OPINIONS")
    logging.info("=" * 70)

    # Load generation checkpoint
    generations_df = pd.read_csv(generation_checkpoint_path, sep="\t", dtype=str, quoting=3)
    logging.info(f"Loaded {len(generations_df)} generated responses")

    # Load ensemble weights
    ensemble_models, _ = load_ensemble_weights(weights_file)

    if len(ensemble_models) == 0:
        raise ValueError(f"No models found in weights file: {weights_file}")

    # Cache key for embeddings (unique to this script/run)
    cache_key = f"{dataset}_{model_sanitized}_{topic}_llm_opinion".replace('.', '_')

    # Run ensemble prediction
    texts = generations_df['generated_text'].tolist()
    predictions, confidences = predict_texts(
        texts, topic, ensemble_models, instruction, centroids_dir,
        embed_batch_size, embeddings_cache_dir, cache_key, device,
        quantification_method=quantification_method,
    )

    # =====================================================================
    # PHASE 3: Save Results
    # =====================================================================
    logging.info("\n" + "=" * 70)
    logging.info("PHASE 3: SAVING RESULTS")
    logging.info("=" * 70)

    # Build output dataframe
    output_df = pd.DataFrame({
        'prompt_id': generations_df['prompt_id'],
        'output_id': generations_df['output_id'],
        'topic': generations_df['topic'],
        'generated_text': generations_df['generated_text'],
        'prediction': predictions,
        'confidence': confidences
    })

    # Save output
    output_df.to_csv(output_path, sep="\t", index=False, quoting=3)
    logging.info(f"Saved predictions to {output_path}")

    # Print final summary
    logging.info("\n" + "=" * 70)
    logging.info("MEASUREMENT COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Model: {model}")
    logging.info(f"Topic: {topic}")
    logging.info(f"Total responses: {len(output_df)}")

    n_for = (predictions == 'for').sum()
    n_against = (predictions == 'against').sum()
    mean_confidence = confidences.mean()

    logging.info(f"Predictions: {n_for} for ({100*n_for/len(output_df):.1f}%), {n_against} against ({100*n_against/len(output_df):.1f}%)")
    logging.info(f"Mean confidence: {mean_confidence:.4f}")
    logging.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
