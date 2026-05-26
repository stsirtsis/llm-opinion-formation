#!/usr/bin/env python3

import os
import random
import logging
import sys

import click
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_prompts, task_dataset_compatible, TASK_ALLOWED_DATASETS, BULLET_TASKS, BULLET_SENTINEL

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
            device_map="auto",
            local_files_only=True
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



def sample_dataframe(df, group_column, sample_per_group, rng):
    """Balanced sampling of rows per group, capped at the smallest group size."""
    if not group_column or sample_per_group <= 0 or len(df) == 0:
        return df

    groups = {group_val: group_df for group_val, group_df in df.groupby(group_column)}
    min_group_size = min(len(g) for g in groups.values())
    n = min(sample_per_group, min_group_size)
    logging.info(f"Balanced sampling: requested {sample_per_group}, smallest group has {min_group_size}, using {n} per group")

    sampled = []
    for group_val, group_df in groups.items():
        sampled_indices = rng.choice(group_df.index, size=n, replace=False)
        sampled.append(group_df.loc[sampled_indices])
        logging.info(f"  Group '{group_val}': sampled {n} rows")

    df_sampled = pd.concat(sampled, ignore_index=True)
    logging.info(f"Sampled {n} per {group_column}: {len(df)} -> {len(df_sampled)} rows")
    return df_sampled



def map_stance_to_text(stance_value):
    """Map stance column values (e.g., 'Argument_for') to natural language (e.g., 'in favor of')."""
    mapping = {
        'argument_for': 'in favor of',
        'argument_against': 'against',
    }
    return mapping.get(stance_value.lower().strip(), stance_value)


def format_prompt(text, topic, stance, user_prompt_template):
    """Format user prompt template with {text}, {topic}, and {stance} placeholders."""
    topic_formatted = topic.replace('_', ' ') if topic else ''
    return user_prompt_template.format(text=text, topic=topic_formatted, stance=stance)


def generate_response(model, tokenizer, model_name, system_prompt, user_prompt, max_new_tokens, temperature, top_p, response_prefix='', log_prompts=False):
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

    if log_prompts:
        logging.info(f"[Chat template output]:\n{input_text}")

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

    # Strip tabs and carriage returns to keep TSV output clean.
    # Newlines are handled by the caller (task-dependent: joined with a space
    # for single-post tasks, preserved via a sentinel for bullet-list tasks).
    response = response.replace('\t', ' ').replace('\r', ' ')

    return response


@click.command()
@click.option('--exp_name', type=str, required=True, help='Experiment name')
@click.option('--output_dir', type=str, required=True, help='Output directory for results')
@click.option('--output_filename', type=str, required=True, help='Output filename (auto-generated by executor)')
@click.option('--dataset', type=str, required=True, help='Dataset name (e.g., ukp, semeval)')
@click.option('--task', type=str, required=True, help='Task name (e.g., writing, improvement) — selects which prompts to use')
@click.option('--model', type=str, required=True, help='HuggingFace model name')
@click.option('--text_column', type=str, required=True, help='Column name containing text to transform')
@click.option('--topic', type=str, default='', help='Topic to filter by (empty = no topic filter)')
@click.option('--topic_column', type=str, default='', help='Column containing topic')
@click.option('--stance_column', type=str, default='', help='Column containing stance (empty = no stance)')
@click.option('--group_column', type=str, default='', help='Column to group by for sampling (empty = no grouping)')
@click.option('--sample_per_group', type=int, default=0, help='Number of samples per group (0 = all rows)')
@click.option('--num_outputs', type=int, default=1, help='Number of transformed versions per input per prompt template')
@click.option('--prompts_file', type=str, default='configs/prompts.json', help='Path to centralized prompts JSON file')
@click.option('--prompts_task', type=str, default='transform_text', help='Task key to look up in prompts JSON file')
@click.option('--max_new_tokens', type=int, default=256, help='Maximum tokens to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature')
@click.option('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--filter_correctly_classified', is_flag=True, default=False, help='Only transform statements correctly classified by the ensemble')
@click.option('--ensemble_dir', type=str, default='models', help='Directory containing ensemble predictions file')
@click.option('--prompt_variant', type=str, default='', help='Prompt variant name (e.g., subtle, identity). Empty = no variant layer.')
@click.option('--log_prompts', is_flag=True, default=False, help='Log the formatted system/user prompts and chat-template output for each generation.')
def main(exp_name, output_dir, output_filename, dataset, task, model, text_column, topic, topic_column,
         stance_column, group_column, sample_per_group, num_outputs,
         prompts_file, prompts_task, max_new_tokens, temperature, top_p, seed,
         filter_correctly_classified, ensemble_dir, prompt_variant, log_prompts):

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
    rng = np.random.default_rng(seed=seed)

    # Load prompts from centralized file
    prompts_config = load_prompts(prompts_file, prompts_task, task, variant=prompt_variant if prompt_variant else None)
    system_prompts = prompts_config['system_prompts']
    prompt_templates = prompts_config['prompts']
    num_system_prompts = len(system_prompts)
    num_user_prompts = len(prompt_templates)
    total_outputs = num_system_prompts * num_user_prompts * num_outputs

    logging.info(f"Experiment: {exp_name}")
    logging.info(f"Model: {model}")
    logging.info(f"System prompts: {num_system_prompts}, user prompt templates: {num_user_prompts}, outputs per template: {num_outputs}, total per input: {total_outputs}")
    logging.info("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load input file
    input_file = os.path.join("data", "processed", f"{dataset}.tsv")
    logging.info(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file, sep="\t", dtype=str, quoting=3, on_bad_lines='warn')

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")

    logging.info(f"Loaded {len(df)} rows")

    # Filter by topic if specified
    if topic:
        df = df[df[topic_column] == topic].copy()
        logging.info(f"Filtered by topic '{topic}': {len(df)} rows")

    # Filter to only correctly classified statements
    if filter_correctly_classified:
        predictions_file = os.path.join(ensemble_dir, f"ensemble_predictions_{dataset}.tsv")
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(
                f"Ensemble predictions file not found: {predictions_file}. "
                "Run build_ensemble first to generate it."
            )
        pred_df = pd.read_csv(predictions_file, sep="\t", quoting=3)
        correct_ids = set(pred_df[pred_df['correctly_classified']]['sentence_id'].astype(int))
        before_len = len(df)
        df = df[df['sentence_id'].astype(int).isin(correct_ids)].copy()
        logging.info(f"Filtered to correctly classified: {before_len} -> {len(df)} rows ({before_len - len(df)} removed)")

    # Apply sampling
    if group_column:
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found. Available: {list(df.columns)}")
        df = sample_dataframe(df, group_column, sample_per_group, rng)

    # Drop rows with missing text
    original_len = len(df)
    df = df.dropna(subset=[text_column])
    if len(df) < original_len:
        logging.info(f"Dropped {original_len - len(df)} rows with missing text")

    # Keep sentence_id, text column, and annotation (for gold-label direction in Bayesian analysis)
    keep_cols = ['sentence_id', text_column]
    if 'annotation' in df.columns:
        keep_cols.append('annotation')
    df = df[keep_cols].copy()
    df = df.reset_index(drop=True)

    # Duplicate rows for multiple outputs and/or multiple prompt templates
    if total_outputs > 1:
        df = df.loc[df.index.repeat(total_outputs)].reset_index(drop=True)
        df['system_prompt_id'] = df.groupby('sentence_id').cumcount() // (num_user_prompts * num_outputs) + 1
        df['prompt_id'] = (df.groupby('sentence_id').cumcount() // num_outputs) % num_user_prompts + 1
        df['output_id'] = df.groupby('sentence_id').cumcount() % num_outputs + 1
        logging.info(f"Duplicated rows for {total_outputs} outputs per input: {len(df)} total rows")
    else:
        df['system_prompt_id'] = 1
        df['prompt_id'] = 1
        df['output_id'] = 1

    # Check for existing checkpoint and filter out completed sentence_ids
    output_path = os.path.join(output_dir, output_filename.replace('.json', '.tsv'))
    completed_ids = set()

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, sep="\t", dtype=str, quoting=3)

        # Count outputs per sentence_id
        output_counts = existing_df.groupby('sentence_id').size()

        # Identify complete vs incomplete sentence_ids
        complete_ids = set(output_counts[output_counts >= total_outputs].index)
        incomplete_ids = set(output_counts[output_counts < total_outputs].index)

        if incomplete_ids:
            logging.info(f"Found {len(incomplete_ids)} incomplete sentence_ids - will regenerate all outputs for these")
            # Remove incomplete rows from checkpoint
            existing_df = existing_df[~existing_df['sentence_id'].isin(incomplete_ids)]
            existing_df.to_csv(output_path, sep="\t", index=False, quoting=3)
            logging.info("Removed incomplete rows from checkpoint")

        completed_ids = complete_ids
        logging.info(f"Resuming from checkpoint: {len(completed_ids)} sentence_ids fully processed")

    if completed_ids:
        df = df[~df['sentence_id'].isin(completed_ids)]

    if len(df) == 0:
        logging.info("All sentence_ids already processed. Nothing to do.")
        return

    total_sentence_ids = df['sentence_id'].nunique()
    logging.info(f"Remaining to process: {len(df)} rows ({total_sentence_ids} sentence_ids)")

    # Load model and tokenizer
    logging.info(f"Loading model: {model}")
    model_obj, tokenizer = load_model_and_tokenizer(model)
    logging.info("Model loaded successfully!")

    # Process by sentence_id groups with checkpointing
    processed_count = 0
    for sentence_id, group in df.groupby('sentence_id', sort=False):
        # Process all rows for this sentence_id
        group_results = []
        for _, row in group.iterrows():

            # Get topic from row if available, otherwise use CLI parameter
            if topic_column and topic_column in df.columns:
                row_topic = row[topic_column] if pd.notna(row[topic_column]) else ''
            else:
                row_topic = topic or ''

            # Get stance
            if stance_column and stance_column in df.columns:
                stance_raw = row[stance_column] if pd.notna(row[stance_column]) else ''
                stance = map_stance_to_text(stance_raw) if stance_raw else ''
            else:
                stance = ''

            # Select system prompt and user prompt template by their ids
            system_prompt = system_prompts[int(row['system_prompt_id']) - 1]
            if '{topic}' in system_prompt:
                system_prompt = system_prompt.format(topic=row_topic.replace('_', ' '))
            prompt_entry = prompt_templates[int(row['prompt_id']) - 1]
            text = row[text_column]
            user_prompt = format_prompt(text, row_topic, stance, prompt_entry['user_prompt'])

            if log_prompts:
                logging.info(
                    f"[Prompts] sentence_id={sentence_id} "
                    f"system_prompt_id={row['system_prompt_id']} prompt_id={row['prompt_id']}\n"
                    f"  [System]: {system_prompt}\n"
                    f"  [User]:   {user_prompt}"
                )

            # Generate response
            response = generate_response(
                model_obj, tokenizer, model, system_prompt, user_prompt,
                max_new_tokens, temperature, top_p, prompt_entry.get('response_prefix', ''),
                log_prompts=log_prompts
            )

            # Flatten newlines for TSV: preserve bullet structure via sentinel for
            # bullet-list tasks, collapse to spaces otherwise.
            if task in BULLET_TASKS:
                response = response.replace('\n', BULLET_SENTINEL)
            else:
                response = response.replace('\n', ' ')

            group_results.append(response)

        # Add results to group and checkpoint
        group = group.copy()
        group['transformed_text'] = group_results

        # Append to checkpoint file (write header only if file doesn't exist)
        header = not os.path.exists(output_path)
        group.to_csv(output_path, sep="\t", index=False, mode='a', header=header, quoting=3)

        processed_count += 1
        logging.info(f"Checkpointed sentence_id {sentence_id} ({processed_count}/{total_sentence_ids})")

    # Print summary (read from file for accurate stats)
    final_df = pd.read_csv(output_path, sep="\t", dtype=str, quoting=3)
    logging.info("\n" + "=" * 70)
    logging.info("TRANSFORMATION COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Input: {input_file}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Total rows in output: {len(final_df)}")
    logging.info(f"Unique original texts: {final_df['sentence_id'].nunique()}")
    logging.info(f"System prompts: {num_system_prompts}, user prompt templates: {num_user_prompts}, outputs per template: {num_outputs}, total per input: {total_outputs}")


if __name__ == "__main__":
    main()
