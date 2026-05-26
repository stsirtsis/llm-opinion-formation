#!/usr/bin/env python3
"""
Download all models needed for the LLM opinion formation experiments.
Most models are cached in models/hf_cache/ under the project root.
Ministral-3 is downloaded to the default HF cache (~/.cache/huggingface/hub)
because MistralCommonBackend does not respect HF_HUB_CACHE.
If a model is already fully cached, does nothing.
"""

import os
import pathlib

_project_root = pathlib.Path(__file__).resolve().parent.parent
_project_cache = str(_project_root / "models" / "hf_cache")

# Models that must go to the default HF cache (MistralCommonBackend ignores HF_HUB_CACHE)
MISTRAL_MODELS = [
    "mistralai/Ministral-3-8B-Instruct-2512",
]

# All other models go to the project cache
LLM_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-8B",
]

EMBEDDING_MODELS = [
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "Salesforce/SFR-Embedding-Mistral",
    "Qwen/Qwen3-Embedding-8B",
]


def get_cached_repos(cache_dir=None):
    from huggingface_hub import scan_cache_dir
    try:
        cache_info = scan_cache_dir(cache_dir)
        return {repo.repo_id for repo in cache_info.repos}
    except Exception:
        return set()


def download(kind, model_id, cache_dir=None):
    from huggingface_hub import snapshot_download
    print(f"[{kind}] {model_id} — downloading...")
    try:
        path = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"  -> Saved to {path}")
    except Exception as e:
        print(f"  -> ERROR: {e}")


def main():
    # Mistral: default HF cache
    mistral_cached = get_cached_repos()
    for model_id in MISTRAL_MODELS:
        if model_id in mistral_cached:
            print(f"[LLM] {model_id} — already cached, skipping.")
        else:
            download("LLM", model_id, cache_dir=None)

    # Everything else: project cache
    os.environ["HF_HUB_CACHE"] = _project_cache
    project_cached = get_cached_repos(_project_cache)
    for kind, model_id in [("LLM", m) for m in LLM_MODELS] + [("Embedding", m) for m in EMBEDDING_MODELS]:
        if model_id in project_cached:
            print(f"[{kind}] {model_id} — already cached, skipping.")
        else:
            download(kind, model_id, cache_dir=_project_cache)


if __name__ == "__main__":
    main()
