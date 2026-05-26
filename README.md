# AI-Mediated Communication Can Steer Collective Opinion

This repository contains the code used in the paper ["AI-Mediated Communication Can Steer Collective Opinion"](https://arxiv.org/abs/2605.16245) by Stratis Tsirtsis, Kai Rawal, Chris Russell, Brent Mittelstadt, and Sandra Wachter.

__Contents__:
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Datasets](#datasets)
- [Language models](#language-models)
- [Running experiments](#running-experiments)
- [Pipeline & repository structure](#pipeline--repository-structure)
- [Attribution](#attribution)

## Introduction

<div align="center">
  <img width="600" src="assets/banner.png">
  <p><sub>Figure designed using resources from <a href="https://www.flaticon.com">flaticon.com</a>.</sub></p>
</div>

Generative artificial intelligence (AI) is increasingly integrated into the online platforms where humans exchange opinions; large language models (LLMs) now polish users' posts on LinkedIn and provide context for content shared on X. There is significant evidence that AI can express biased opinions and shape individuals' opinions during human-AI interactions, however, less attention has been paid to its influence on collective opinion formation when mediating human-to-human communication. Our work addresses this gap via a combination of empirical and theoretical analyses. We show empirically that LLMs from multiple popular families introduce directional biases when instructed to edit human-written texts on contested topics, for example, nudging texts in favor of gun control and against atheism. Building on this observation, we introduce a mathematical model of opinion dynamics in which an AI system sits between users on a social network, transforming the opinions they express and perceive. By analytically characterizing the equilibrium of this model and performing simulations on real social network data, we show that biases introduced by AI in human-to-human communication can be amplified through the network and shift collective opinion in their direction. In light of these findings, we investigate whether such biases are controllable by online platforms. We audit the "Explain this post" feature on X and find evidence of pro-life bias in Grok's outputs on abortion-related content, which we trace back to specific design choices.


## Dependencies

### Python
All experiments in Python were performed using version 3.13.11. To create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```


### Julia
All experiments in Julia were performed using version 1.12.6. To install dependencies, open the Julia REPL and run:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
or equivalently
```julia
] activate .
] instantiate
```

### R

All experiments in R were performed using version 4.3.1. To install dependencies, open the R console and run:

```r
install.packages(c("brms", "tidyverse", "posterior", "jsonlite"))
```


## Datasets

- **[UKP](https://tudatalib.ulb.tu-darmstadt.de/items/18c6544d-fa61-4505-ae8f-1304352c06e6)** — arguments for/against 8 topics (abortion, cloning, death penalty, gun control, marijuana legalization, minimum wage, nuclear energy, school uniforms).
- **[SemEval](https://alt.qcri.org/semeval2016/task6/)** — tweets for/against 6 topics (atheism, climate action, feminism, Hillary Clinton, abortion, Donald Trump).
- **[twitter, gplus, facebook](https://snap.stanford.edu/data/#socnets)** — social network graphs from the Stanford Large Network Dataset Collection (SNAP).

The human-written texts contained in both datasets are shared publicly on the web. However, their collection, labeling, and preprocessing were performed by third parties. Therefore, we do not include the raw data in this repository, and we point to the original sources instead (see links above). To run our experiments, one needs to get the raw data from their original sources and place them under `data/original/UKP_sentential_argument_mining/` and `data/original/semeval-data-all-annotations/`. Then, our two scripts `src/preprocess_ukp.py` and `src/preprocess_semeval.py` use that raw data to create TSV files under `data/processed/`, which are then used for our experiments. Specifically, `src/preprocess_ukp.py` reads the `[TOPIC].tsv` files contained in the raw UKP dataset  and `src/preprocess_semeval.py` reads the files `trainingdata-all-annotations.txt`, `testdata-taskA-all-annotations.txt`, `testdata-taskB-all-annotations.txt`, `trialdata-all-annotations.txt` contained in the raw SemEval dataset. Then, they produce the processed versions `data/processed/ukp.tsv` and `data/processed/semeval.tsv`.

In `data/few_shot/judge_abortion_hash.tsv`, we include the hash values of the texts from the UKP dataset that we used as few-shot examples to guide the judge model when classifying claims about abortion. In `data/processed/sem_eval_active_x_links.tsv`, we include the current X links of the tweets from the SemEval dataset that we used to audit X's "Explain this post" feature. Our code implementing the experiment with Grok (see `src/grok_explain.py`) uses a tsv file named `data/processed/semeval_active_x_posts.tsv` that contains the fields `sentence_id`, `text`, `annotation`, `link`, which we created manually by using the texts and class annotations of tweets on abortion from SemEval.

## Language models

### Generation models

Used to draft, improve, or contextualize human-written posts.

- mistralai/Ministral-3-8B-Instruct-2512 (Hugging Face)
- meta-llama/Llama-3.1-8B-Instruct (Hugging Face)
- google/gemma-3-12b-it (Hugging Face)
- Qwen/Qwen3-8B (Hugging Face)
- grok-4-1-fast-reasoning (xAI API)

### Embedding models

Used to build the ensemble that classifies text as in favor of or against each topic. All models are available on Hugging Face.

- Qwen/Qwen3-Embedding-8B (4096 dim)
- tencent/KaLM-Embedding-Gemma3-12B-2511 (3840 dim)
- Salesforce/SFR-Embedding-Mistral (4096 dim)
- Octen/Octen-Embedding-8B (4096 dim)
- Linq-AI-Research/Linq-Embed-Mistral (4096 dim)

### Judge model

We use gpt-5.4 (via OpenAI's API) to label contextual claims generated by Grok.


## Running experiments

Each Python/Julia script is treated as an "experiment" whose parameters are configured by a JSON file in `configs/`.

### How configs work

Each config file contains three types of entries:

- **Scalar values** — fixed parameters shared across all runs (e.g., `"seed": 42`).
- **Array values** — variable parameters that form a grid (e.g., `"model": ["llama", "gemma"]`, `"topic": ["abortion", "cloning"]`).
- **Reserved keys** — `executor` (script name, output/log dirs) and `slurm` (cluster resources); these are not passed as experiment parameters.

`src/executor.py` takes the Cartesian product of all array-valued parameters to generate the set of runs. For each run it invokes `src/{exec_name}.py` (or `.jl`) with all parameters passed as command-line arguments. Output filenames are auto-generated from the variable parameters: `{name}__{key1}={val1}__{key2}={val2}.tsv`. Outputs are saved to `outputs/{experiment_name}/` (configurable via `executor.output_dir`). Logs go to `outputs/logs/{experiment_name}/` and SLURM messages go to `outputs/slurm_logs/`.

### Running locally

```bash
source env/bin/activate
bash scripts/run.sh -n transform_text
bash scripts/run.sh -n shift -l julia
```

This iterates sequentially over all parameter combinations in the config.

### Submitting to SLURM

```bash
bash scripts/submit_slurm.sh -n transform_text
bash scripts/submit_slurm.sh -n transform_text -d
```

This submits a SLURM array job with one task per parameter combination. Resources (partition, memory, GPUs, etc.) are read from the `slurm` section of the config. The `-d` flag prints the `sbatch` command without scheduling any jobs.

### Bayesian analysis (R)

R scripts are run directly rather than through the executor:

```bash
Rscript src/run_bayesian_analysis.R
Rscript src/run_bayesian_analysis_steer.R
Rscript src/run_bayesian_grok.R
Rscript src/run_bayesian_grok_guideline.R
```



## Pipeline & repository structure

### Measuring directional biases in AI-mediated opinion expression (Section 2)

| Step | Script | Config | Description |
|------|--------|--------|-------------|
| 1. Preprocess | `src/preprocess_ukp.py`, `src/preprocess_semeval.py` | None (run directly) | Convert raw datasets to a shared format under `data/processed/{dataset}.tsv` |
| 2. Build ensemble | `src/build_ensemble.py` | `configs/build_ensemble.json` | Build a classifier using 5 embedding models to assign an opinion score to a text |
| 3. Transform text | `src/transform_text.py` | `configs/transform_text.json` | Use an LLM to draft/improve posts with prompts from `configs/prompts.json` |
| 4. Predict opinion | `src/predict_opinion.py` | `configs/predict_opinion.json` | Use the ensemble to assign opinion scores to both original and transformed texts |
| 5. Measure LLM opinion | `src/measure_llm_opinion.py` | `configs/measure_llm_opinion.json` | Generate free-form statements to measure each LLM's directly expressed opinion per topic |
| 6. Bayesian analysis | `src/run_bayesian_analysis.R` | None (run directly) | Fit Bayesian linear mixed-effects models to quantify bias introduced each LLM on each topic |

**Dependencies:** Step 2 must run before steps 4 and 5. Step 3 must run before step 4. Step 4 must run before step 6.

### Prepending ideological viewpoints to system prompts (Appendix E)

Steps must be run in order.

| Step | Script | Config | Description |
|------|--------|--------|-------------|
| Steer–transform | `src/transform_text.py` | `configs/steer_prompt_transform.json` | Repeat the same transformation tasks as beforte under 7 political prefix system prompts |
| Steer–predict   | `src/predict_opinion.py` | `configs/steer_prompt_predict.json` | Assign opinion scores to the steered outputs using the ensemble |
| Steer–Bayesian  | `src/run_bayesian_analysis_steer.R` | None (run directly) | Fit per-prefix mixed-effects models on the resulting opinion shifts |

### Auditing X's "Explain this post" feature (Section 4)

Steps must be run in order.

| Step | Script | Config | Description |
|------|--------|--------|-------------|
| Generate contextual claims    | `src/grok_explain.py` | `configs/grok_explain.json` | Call the Grok API with the contextualization prompt; `exclude_guideline=0` includes all four guidelines by X; `exclude_guideline` ∈ {1,2,3,4} excludes one guideline at a time |
| Judge contextual claims       | `src/judge_predict.py` | `configs/judge_predict.json` | Classify each contextual claim's stance (`for`/`against`/`neutral`) using a few-shot LLM-as-a-judge approach with OpenAI's gpt-5.4 |
| Bayesian baseline   | `src/run_bayesian_grok.R` | None (run directly) | Fit a Bayesian categorical mixed-effects model to test for bias in Grok's contextual claims under the full set of guidelines |
| Bayesian guideline analysis   | `src/run_bayesian_grok_guideline.R` | None (run directly) | Fit a Bayesian categorical mixed-effects model across guideline conditions to identify which guideline drives the bias |

### Simulating opinion dynamics using real network data and AI transformations (Section 3.2)

All experiments use the script `src/simulation.jl` and require that Step 6 mentioned above (Section 2 - Bayesian analysis) is complete. We specify three configs to perform three different types of analyses.

| Config | Purpose |
|--------|---------|
| `dynamics.json`  | Convergence analysis (Appendix D) and average opinion shift as a function of the fraction of AI adopters |
| `shift.json`     | Full sweep across all topics to analyze the correlation between opinion shift and bias introduced by the respective transformation |
| `fj_params.json` | Sensitivity to community ratio and stubbornness mean across networks and topics |

## Repository structure

```
llm-opinion-formation/
├── configs/
│   ├── prompts.json                     # Centralized prompt templates for all tasks
│   ├── build_ensemble.json              # Step 2: ensemble construction
│   ├── transform_text.json              # Step 3: editing task
│   ├── predict_opinion.json             # Step 4: ensemble classification
│   ├── measure_llm_opinion.json         # Step 5: directly expressed opinion measurement
│   ├── steer_prompt_transform.json      # Steering experiment: rewrite under political prefixes
│   ├── steer_prompt_predict.json        # Steering experiment: assign opinion scores to steered outputs
│   ├── grok_explain.json                # Contextualization experiment: generate claims
│   ├── judge_predict.json               # Contextualization experiment: judge claim stance
│   ├── shift.json                       # Simulation: full topic sweep
│   ├── dynamics.json                    # Simulation: convergence analysis and opinion shift vs fraction of AI adopters
│   └── fj_params.json                   # Simulation: community-ratio × stubbornness sweep
├── data/
│   ├── original/                        # Raw datasets and social-network files (not redistributed; obtain from the original sources)
│   ├── processed/                       # Preprocessed TSVs
│   └── few_shot/                        # Few-shot examples for the judge model
├── src/
│   ├── executor.py                      # Expands config grids and dispatches Python/Julia scripts
│   ├── utils.py                         # Shared utilities
│   ├── preprocess_ukp.py                # Step 1: preprocess UKP
│   ├── preprocess_semeval.py            # Step 1: preprocess SemEval
│   ├── build_ensemble.py                # Step 2: build ensemble classifier
│   ├── transform_text.py                # Step 3: LLM text transformation
│   ├── predict_opinion.py               # Step 4: ensemble classification
│   ├── measure_llm_opinion.py           # Step 5: directly expressed opinion measurement
│   ├── run_bayesian_analysis.R          # Step 6: Bayesian mixed-effects model on the main shifts
│   ├── run_bayesian_analysis_steer.R    # Steering experiment: per-prefix Bayesian models
│   ├── simulation.jl                    # Opinion dynamics simulation
│   ├── grok_explain.py                  # Contextualization: generate claims via Grok API
│   ├── judge_predict.py                 # Contextualization: judge claim stance via OpenAI API
│   ├── run_bayesian_grok.R              # Contextualization: baseline bias model
│   ├── run_bayesian_grok_guideline.R    # Contextualization: joint model for guideline analysis
│   └── cross_dataset_eval.py            # Cross-dataset evaluation of centroid transfer using abortion (shared topic)
├── scripts/
│   ├── run.sh                           # Run experiments locally
│   ├── submit_slurm.sh                  # Submit SLURM array jobs
│   ├── clean.sh                         # Clean log files
│   ├── build_sysimage.jl                # Precompile Julia packages into a system image for faster loading
│   └── cache_hf_models.py               # Download all required Hugging Face models to the project cache
├── notebooks/
│   ├── bias.ipynb                       # Opinion-shift visualizations (main pipeline)
│   ├── transformation.ipynb             # Fits LLM transformation function via kernel regression; writes outputs used by simulation
│   ├── steer_prompt.ipynb               # Steering experiment results
│   ├── grok_bias.ipynb                  # Contextualization experiment results
│   ├── shift.ipynb                      # Simulation: long-run average opinion vs bias intercept across topics and networks
│   ├── dynamics.ipynb                   # Simulation: convergence analysis and opinion shift vs fraction of AI adopters
│   └── fj_params.ipynb                  # Simulation: community-ratio x stubbornness sweep
├── models/                              # Ensemble weights, centroids, embedding caches
├── outputs/                             # Experiment outputs, organized by experiment name
└── figures/                             # Experiment figures (produced by notebooks)
```

## Attribution

If you use this code in your research, please cite:

```bibtex
@article{tsirtsis2026aimediated,
      title={AI-Mediated Communication Can Steer Collective Opinion}, 
      author={Stratis Tsirtsis and Kai Rawal and Chris Russell and Brent Mittelstadt and Sandra Wachter},
      journal={arXiv preprint arXiv:2605.16245},
      year={2026},
}
```
