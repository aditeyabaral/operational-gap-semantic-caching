# Closing the Operational Gap in Semantic Caching

Official code for the paper 📄 **"Closing the Operational Gap in Semantic Caching."** [[arXiv]](https://arxiv.org/abs/2606.19719)

🤗 **Models and datasets:** [`redis` on HuggingFace](https://huggingface.co/redis)

If you use this code, the models, or the datasets, please cite:

```bibtex
@misc{baral2026closingoperationalgapsemantic,
      title={Closing the Operational Gap in Semantic Caching},
      author={Aditeya Baral and Radoslav Ralev and Iliya Sotirov Zhechev and Srijith Rajamohan and Jen Agarwal},
      year={2026},
      eprint={2606.19719},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2606.19719},
}
```

Semantic caching cuts LLM inference costs by serving a cached response when a new query is *semantically* similar to a previously seen one. Whether a cache should fire is decided by a **score threshold**, yet semantic-cache retrievers and re-rankers are almost always selected by **PR-AUC** — a threshold-agnostic metric that says nothing about whether those scores are *usable at a fixed threshold*. This repository contains everything needed to reproduce our study of that mismatch: dataset curation, re-ranker fine-tuning, end-to-end retrieval + re-ranking evaluation against a live [Redis](https://redis.io/) semantic cache, and the cache-aware analysis that quantifies the **operational (threshold-utility) gap**.

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Models and Datasets](#models-and-datasets)
- [Reproducing the Paper](#reproducing-the-paper)
- [Detailed Usage](#detailed-usage)
  - [1. Dataset Curation](#1-dataset-curation)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Analysis](#4-analysis)

## Overview

Standard practice evaluates semantic-cache retrievers and re-rankers with **PR-AUC**, a threshold-agnostic metric (agnostic to score magnitude) that ignores whether scores are usable at a fixed operating threshold. We show this leads to systematically poor deployment choices — **the models with the highest PR-AUC are often the worst in operation.**

The paper introduces two cache-aware tools and a decomposition:

- **Precision–Cache Hit Ratio (P-CHR) AUC** — precision measured across the full range of cache *utilization* levels, rather than recall. Computed **exactly (grid-free)**, by sweeping the threshold through every distinct score, so it is invariant to any monotone rescaling and commensurable across score semantics.
- **Operational Retention Rate (ORR)** — how much of a model's offline ranking quality actually survives at deployment.
- A decomposition of the offline-to-deployed quality gap into a **recoverable threshold-utility component** and an **irreducible structural component** fixed by the dataset's positive rate.

Our experiments show the threshold-utility gap is governed by the **training objective** (binary cross-entropy vs. multiple-negatives ranking loss) rather than data scale, and that post-hoc calibration cannot close it — a monotone rescaling leaves the rank-invariant metric unchanged, so the gap yields only to re-normalizing scores over the candidate pool or changing the objective. The central takeaway: **model selection for semantic caching is a threshold-utility problem, not a ranking one.**

This repository lets you reproduce that result end-to-end and apply the same analysis to your own retrievers and re-rankers.

## Key Concepts

These definitions make the repository self-contained; see the paper for full treatment.

| Term | Definition |
|------|------------|
| **Semantic cache** | A store of past query→response pairs. A new query is embedded, the nearest cached entries are retrieved, and a re-ranker scores them; if the top score clears a threshold, the cached response is served (a **cache hit**). |
| **Retriever (bi-encoder)** | Embedding model that fetches candidate matches from the cache by vector similarity. |
| **Re-ranker** | Cross-encoder or ColBERT model that re-scores the retrieved candidates more precisely. |
| **Cache Hit Ratio (CHR)** | Fraction of queries for which the cache fires (top score ≥ threshold), regardless of correctness. |
| **Valid Cache Hit Ratio (VCHR)** | Fraction of queries where the cache fires **and** returns the correct match (= precision × CHR). |
| **PR-AUC** | Area under the precision–recall curve, swept over the decision threshold and reported for both retrievers and re-rankers. It is agnostic to score magnitude, so a high PR-AUC need not translate into good precision at a fixed deployment threshold. |
| **P-CHR AUC** | Area under the precision-vs-CHR curve — precision across operating points defined by *how much* of the cache is used. Computed exactly (grid-free), so it is rank-invariant and commensurable across score semantics. The paper's recommended selection metric. |
| **Operational Retention Rate (ORR)** | P-CHR AUC / PR-AUC — how much offline ranking quality is retained at a deployable fixed threshold. |
| **Operational gap** | PR-AUC − P-CHR AUC: the offline→deployed quality drop, split into the two parts below. |
| **Threshold-utility gap** | The recoverable part, from where a model places its scores relative to the threshold. Post-hoc temperature / Platt scaling **cannot** close it (monotone rescaling leaves the rank-invariant metric unchanged); re-normalizing scores over the candidate pool or changing the training objective can. |
| **Structural gap** | The irreducible part, fixed by the dataset's positive rate. |

## Repository Structure

```
.
├── src/
│   ├── analysis/
│   │   ├── analyze_cls.py              # PR-AUC, exact P-CHR/P-VCHR AUC, ORR + gap decomposition, curves
│   │   ├── analyze_distribution.py    # Score distribution (KDE) + ECE/NLL/Brier + correlations
│   │   ├── analyze_latency.py         # Retrieval and reranking latency analysis and plots
│   │   ├── compute_calibration.py     # Temperature/Platt fitting + calibration-parameter table
│   │   ├── dataset_stats.py           # Per-version dataset split statistics
│   │   └── util.py                     # Shared utilities (library module)
│   ├── eval/
│   │   ├── eval_reranker.py            # End-to-end retrieval + re-ranking evaluation
│   │   └── retrieve_rerank_evaluator.py  # Evaluator class (library module)
│   ├── reranker/
│   │   ├── cache_evaluator.py          # Cache-aware SentenceEvaluator (library module)
│   │   ├── finetune_colbert.py         # ColBERT re-ranker fine-tuning script
│   │   ├── finetune_crossencoder.py    # Cross-encoder re-ranker fine-tuning script
│   │   └── util.py                     # Dataset loading and InfoNCE utilities (library module)
│   ├── sentencepairs/
│   │   ├── create_sentencepairs_v1.py  # Dataset curation: v1 (1M train pairs)
│   │   ├── create_sentencepairs_v2.py  # Dataset curation: v2 (8M train pairs)
│   │   └── create_sentencepairs_v3.py  # Dataset curation: v3 (40M train pairs)
│   └── shell/
│       ├── run_reranker_evals.sh               # Run all retriever–reranker eval combinations
│       └── run_reranker_evals_for_retriever.sh # Run all rerankers for one retriever
├── results/                            # Evaluation result JSON files (created by eval_reranker.py)
├── plots/                              # Analysis plots (created by the analysis scripts)
├── pyproject.toml
└── uv.lock
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and targets **Python 3.12**.

```bash
# Clone the repository
git clone https://github.com/aditeyabaral/operational-gap-semantic-caching.git
cd operational-gap-semantic-caching

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into a local virtual environment
uv sync

# Activate the environment
source .venv/bin/activate
```

All scripts are run from the **repository root** (each script adds the root to `sys.path`, so imports resolve as `src.<module>`).

### Redis

Evaluation runs against a live semantic cache backed by **Redis Stack**. The easiest way to start one locally is with Docker:

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

By default the scripts connect to `localhost:6379` (configurable via `--redis-host` / `--redis-port`).

### Hugging Face

A Hugging Face account and access token are required to download the models/datasets and (for training/curation) to push artifacts to the Hub. Authenticate once:

```bash
huggingface-cli login
```

### Hardware

- **Evaluation & analysis:** a single GPU is recommended; CPU works but is slow. The analysis scripts are CPU-parallel.
- **Training:** a CUDA GPU is required. All training scripts support single- and multi-GPU execution via `accelerate launch` (run `accelerate config` once to set up), and use **FlashAttention-2** with `bfloat16`.

## Models and Datasets

All models and datasets introduced in the paper are published under the [`redis`](https://huggingface.co/redis) organization on the Hugging Face Hub. The evaluation suite also includes a set of third-party baselines, which are downloaded automatically.

### Our artifacts

| Type | Hugging Face ID |
|------|-----------------|
| Retriever (bi-encoder) | [`redis/langcache-embed-v1`](https://huggingface.co/redis/langcache-embed-v1) |
| Retriever (bi-encoder) | [`redis/langcache-embed-v2`](https://huggingface.co/redis/langcache-embed-v2) |
| Retriever (bi-encoder) | [`redis/langcache-embed-v3-small`](https://huggingface.co/redis/langcache-embed-v3-small) |
| Cross-encoder re-ranker | [`redis/langcache-reranker-v1-bce`](https://huggingface.co/redis/langcache-reranker-v1-bce) |
| Cross-encoder re-ranker | [`redis/langcache-reranker-v1-mnrl`](https://huggingface.co/redis/langcache-reranker-v1-mnrl) |
| Cross-encoder re-ranker (BCE) | [`redis/langcache-reranker-v2-bce`](https://huggingface.co/redis/langcache-reranker-v2-bce) |
| Cross-encoder re-ranker (MNRL) | [`redis/langcache-reranker-v2-mnrl`](https://huggingface.co/redis/langcache-reranker-v2-mnrl) |
| Sentence-pair datasets | [`redis/langcache-sentencepairs-v1`](https://huggingface.co/datasets/redis/langcache-sentencepairs-v1) · [`v2`](https://huggingface.co/datasets/redis/langcache-sentencepairs-v2) · [`v3`](https://huggingface.co/datasets/redis/langcache-sentencepairs-v3) |
| LLM paraphrase source | [`redis/llm-paraphrases`](https://huggingface.co/datasets/redis/llm-paraphrases) |

### Baselines evaluated

- **Retrievers:** `BAAI/bge-base-en-v1.5`, `Alibaba-NLP/gte-modernbert-base`, `jinaai/jina-embeddings-v2-base-en`, `nomic-ai/nomic-embed-text-v1.5`, `intfloat/e5-base-v2`, `Snowflake/snowflake-arctic-embed-m-v2.0`
- **Cross-encoder re-rankers:** `cross-encoder/ms-marco-MiniLM-L12-v2`, `Alibaba-NLP/gte-reranker-modernbert-base`
- **ColBERT re-rankers:** `lightonai/ColBERT-Zero`, `lightonai/GTE-ModernColBERT-v1`, `lightonai/Reason-ModernColBERT`, `colbert-ir/colbertv2.0`

### SentencePairs dataset versions

Three cumulative versions of the sentence-pair dataset are provided, each building on the previous:

| Version | Train | Val | Test | Sources |
|---------|------:|----:|-----:|---------|
| [v1](https://huggingface.co/datasets/redis/langcache-sentencepairs-v1) | 1M | 8.4K | 62K | APT, PAWS, QQP, SICK, STS-B, MRPC, PARADE, PIT-2015 |
| [v2](https://huggingface.co/datasets/redis/langcache-sentencepairs-v2) | 8M | 8.4K | 72K | v1 + LLM-generated paraphrases |
| [v3](https://huggingface.co/datasets/redis/langcache-sentencepairs-v3) | 40M | 10.8K | 74K | v2 + OpusParcus, TTIC-31190, TaPaCo, Paraphrase Collections, ChatGPT Paraphrases, ParaNMT-5M, Task275-WSC, ParaBank2 |

All three versions are pre-built and published on the Hub (linked above), so you do **not** need to rebuild them to reproduce results — they are loaded automatically by the training and evaluation scripts. The paper uses **v3**.

## Reproducing the Paper

The full pipeline is **Datasets → Training → Evaluation → Analysis**. Because the datasets and models are already on the Hub, most users can skip straight to **Evaluation**.

**Quickstart (single combination):**

```bash
# 1. Start Redis and authenticate with Hugging Face
docker run -d -p 6379:6379 redis/redis-stack:latest
huggingface-cli login

# 2. Evaluate one retriever + re-ranker pair at k=50 (the paper's setting)
python src/eval/eval_reranker.py \
  --biencoder-model-path redis/langcache-embed-v3-small \
  --reranker-model-path redis/langcache-reranker-v2-bce \
  --reranker-type crossencoder \
  --dataset-version v3 \
  --top-k 50 \
  --flush-cache \
  --output results/example.json

# 3. Fit post-hoc calibration (temperature / Platt) for the re-ranker
python src/analysis/compute_calibration.py \
  --model-path redis/langcache-reranker-v2-bce \
  --dataset-version v3 \
  --output calibration_params.json

# 4. Analyze: PR-AUC and P-CHR-AUC, with calibration applied
python src/analysis/analyze_cls.py \
  --results-dir results/ \
  --plots-dir plots/classification/ \
  --output plots/classification/cls_metrics.json \
  --calibration calibration_params.json \
  --calibration-method temperature
```

**Full reproduction (all tables and figures).** Every value in the paper is produced by running the
pipeline below on the full set of evaluation results at **`k=50`** (the paper's setting — set
`TOP_K=50` in `src/shell/run_reranker_evals.sh`, whose default is `k=5`). All analysis scripts print
their results as tables and write a JSON; no precomputed data is shipped.

```bash
# 0. Evaluate every retriever × re-ranker combination at k=50. Requires Redis running.
TOP_K=50 bash src/shell/run_reranker_evals.sh          # -> results/*.json (raw eval)

# 1. Metrics: PR-AUC + grid-free exact P-CHR / P-VCHR AUC, ORR, and the operational-gap
#    decomposition. Prints the full per-combo tables, the per-retriever baselines, and the
#    per-reranker averages over retrievers.  [Tables: retriever-baselines, operational-gap,
#    full PR-AUC / P-CHR / P-VCHR; Figure: PR & P-CHR curves]
python src/analysis/analyze_cls.py --results-dir results/ \
  --plots-dir plots/classification/ --output plots/classification/cls_metrics.json

# 2. Score-normalization ablation: exact P-CHR under the counterfactual transform.
#    [Table: score-normalization ablation]  (native run is step 1; run the two counterfactuals)
python src/analysis/analyze_cls.py --results-dir results/ --transform pool_softmax \
  --plots-dir plots/ablation_softmax/ --output plots/ablation_softmax/cls_metrics.json
python src/analysis/analyze_cls.py --results-dir results/ --transform raw_sigmoid \
  --plots-dir plots/ablation_sigmoid/ --output plots/ablation_sigmoid/cls_metrics.json

# 3. Score distributions: ROC-AUC/KS/overlap, ECE/NLL/Brier, KDE plots, and the pooled
#    Spearman correlations of the calibration metrics vs exact P-CHR AUC.
#    [Table: distribution metrics; Figure: KDE panels]
python src/analysis/analyze_distribution.py --results-dir results/ \
  --plots-dir plots/distribution/ --output plots/distribution/dist_metrics.json \
  --cls-metrics plots/classification/cls_metrics.json

# 4. Overhead latency, joined with exact P-CHR AUC and ΔRet.  [Table: latency]
python src/analysis/analyze_latency.py --results-dir results/ \
  --plots-dir plots/latency/ --output plots/latency/latency_metrics.json \
  --cls-metrics plots/classification/cls_metrics.json

# 5. Post-hoc calibration parameters (T, Platt a/b) + ΔP-CHR AUC.  [Table: calib-params]
#    First fit params per BCE/MNRL reranker (GPU), then report the table (no GPU):
python src/analysis/compute_calibration.py --model-path <reranker> --dataset-version v3 \
  --output calibration_params.json                                   # repeat per reranker
python src/analysis/analyze_cls.py --results-dir results/ --calibration calibration_params.json \
  --plots-dir plots/calibrated/ --output plots/calibrated/cls_metrics.json
python src/analysis/compute_calibration.py --report --output calibration_params.json \
  --native-cls-metrics plots/classification/cls_metrics.json \
  --calibrated-cls-metrics plots/calibrated/cls_metrics.json

# 6. Dataset statistics.  [Table: dataset-stats]
python src/analysis/dataset_stats.py
```

> **K-sensitivity** (Table: K-sensitivity) is produced by step 1 as well: `analyze_cls.py` sweeps
> `k=1..K` internally and `plots/classification/pchr_auc_vs_k.png` plus the per-combo tables report
> exact P-CHR AUC at every `k`. The per-source dataset table (all-sources) is a provenance table from
> the curation scripts in `src/sentencepairs/`.

> **Cache reuse.** Populating the cache for a retriever is the expensive step. Pass `--flush-cache` on the **first** evaluation for each new retriever, then omit it for subsequent re-rankers that share the same retriever so the cached embeddings are reused. The shell scripts handle this automatically.

## Detailed Usage

### 1. Dataset Curation

> Optional — only needed to rebuild or extend the datasets from scratch; the pre-built versions are on the Hub.

Each script curates one dataset version, pushes each source split individually, then pushes a merged `all` config to the Hub under your configured account.

```bash
python src/sentencepairs/create_sentencepairs_v1.py
python src/sentencepairs/create_sentencepairs_v2.py
python src/sentencepairs/create_sentencepairs_v3.py
```

Most sources download automatically from the Hub. A few must be placed under a `data/` directory at the repo root beforehand:

**Required for v1, v2 and v3:**

| Source | Expected path | Files |
|--------|--------------|-------|
| PARADE | `data/parade/` | `PARADE_train.txt`, `PARADE_validation.txt`, `PARADE_test.txt` |
| PIT-2015 | `data/pit2015/` | `train.data`, `dev.data`, `test.data` |
| APT | `data/apt/` | `train.tsv`, `test.tsv` |
| SICK | `data/sick/` | `SICK.txt` |

**Additional sources for v3:**

| Source | Expected path | Files |
|--------|--------------|-------|
| TTIC-31190 | `data/ttic31190/` | `train.tsv`, `dev.tsv`, `devtest.tsv` |
| OpusParcus | `data/opusparcus/` | `train_en.70.jsonl`, `validation.jsonl`, `test.jsonl` |
| ParaNMT-5M | `data/paranmt/para-nmt-5m-processed/` | `para-nmt-5m-processed.txt` |
| ParaBank2 | `data/parabank2/` | `parabank2.tsv` |

All other sources (PAWS, MRPC, QQP, STS-B, TaPaCo, Paraphrase Collections, ChatGPT Paraphrases, Task275-WSC, and the LLM paraphrases) are pulled from the Hub automatically.

### 2. Training

> Optional — the trained re-rankers are on the Hub. Training requires a CUDA GPU and supports multi-GPU via `accelerate`.

#### Cross-Encoder Fine-tuning

```bash
accelerate launch src/reranker/finetune_crossencoder.py \
  --pretrained-model-path Alibaba-NLP/gte-reranker-modernbert-base \
  --finetuned-model-path <your-hf-username>/my-reranker \
  --dataset-version v3 \
  --loss-function bce \
  --epsilon 0.5 \
  --batch-size 48 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --output-dir /path/to/checkpoints
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained-model-path` | `Alibaba-NLP/gte-reranker-modernbert-base` | Base cross-encoder to fine-tune |
| `--finetuned-model-path` | `redis/langcache-reranker-v2-bce` | Output model name / Hub ID (also the push target — override with your own namespace) |
| `--dataset-version` | `v3` | SentencePairs version (`v1`, `v2`, `v3`) |
| `--train-dataset-subsets` | `["all"]` | Subset names within the dataset version |
| `--loss-function` | *(required)* | `bce`, `mnrl-sampled`, `mnrl-positive`, or `mse` |
| `--epsilon` | `0` | Label-smoothing coefficient for BCE |
| `--num-negatives` | `1` | Negatives per anchor (MNRL) |
| `--scale` | `20.0` | Scale for MNRL |
| `--batch-size` | `48` | Per-device train/eval batch size |
| `--learning-rate` | `2e-4` | Peak learning rate |
| `--epochs` | `5` | Training epochs |
| `--warmup-ratio` | `0.10` | LR warmup fraction |
| `--weight-decay` | `0.001` | AdamW weight decay |
| `--eval-split` | `val` | Split used for checkpoint selection |
| `--combine-train-and-val` | `False` | Merge train+val into the training set |
| `--eval-steps` / `--save-steps` / `--logging-steps` | `1000` / `10000` / `1000` | Step intervals |
| `--save-total-limit` | `5` | Max checkpoints to keep |
| `--output-dir` | `/opt/dlami/nvme/langcache-reranker-models` | Local checkpoint directory |
| `--wandb-run-name` | `None` | Optional Weights & Biases run name |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

The best checkpoint (by validation F1) is pushed to `--finetuned-model-path` at the end of training.

#### ColBERT Fine-tuning

```bash
accelerate launch src/reranker/finetune_colbert.py \
  --pretrained-model-path lightonai/GTE-ModernColBERT-v1 \
  --finetuned-model-path <your-hf-username>/my-colbert \
  --dataset-version v3 \
  --num-negatives 1 \
  --temperature 0.02 \
  --batch-size 48 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --output-dir /path/to/checkpoints
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained-model-path` | `lightonai/GTE-ModernColBERT-v1` | Base ColBERT model to fine-tune |
| `--finetuned-model-path` | `redis/langcache-colbert-v2` | Output model name / Hub ID (push target — override with your own namespace) |
| `--query-length` / `--document-length` | `512` / `512` | Max query / document token lengths |
| `--dataset-version` | `v3` | SentencePairs version (`v1`, `v2`, `v3`) |
| `--train-dataset-subsets` | `["all"]` | Subset names within the dataset version |
| `--num-negatives` | `1` | Negatives per anchor (contrastive / InfoNCE) |
| `--temperature` | `0.02` | InfoNCE temperature |
| `--batch-size` | `48` | Per-device train/eval batch size |
| `--learning-rate` | `2e-4` | Peak learning rate |
| `--epochs` | `5` | Training epochs |
| `--warmup-ratio` | `0.10` | LR warmup fraction |
| `--weight-decay` | `0.001` | AdamW weight decay |
| `--eval-split` | `val` | Split used for checkpoint selection |
| `--combine-train-and-val` | `False` | Merge train+val into the training set |
| `--eval-steps` / `--save-steps` / `--logging-steps` | `1000` / `10000` / `1000` | Step intervals |
| `--save-total-limit` | `5` | Max checkpoints to keep |
| `--output-dir` | `/opt/dlami/nvme/langcache-colbert-models` | Local checkpoint directory |
| `--wandb-run-name` | `None` | Optional Weights & Biases run name |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

### 3. Evaluation

`eval_reranker.py` runs the full two-stage pipeline against a live Redis semantic cache for **one** retriever–reranker pair and writes a result JSON to `results/`. The cache is populated with the test set's unique candidates, then every test query is retrieved (`top-k`) and re-ranked. Raw re-ranker scores are stored as-is; activation and calibration are applied later, at analysis time.

```bash
python src/eval/eval_reranker.py \
  --biencoder-model-path redis/langcache-embed-v3-small \
  --reranker-model-path redis/langcache-reranker-v2-bce \
  --reranker-type crossencoder \
  --dataset-version v3 \
  --top-k 50 \
  --flush-cache \
  --output results/example.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--biencoder-model-path` | *(required)* | Retriever (bi-encoder) Hub ID or local path |
| `--reranker-model-path` | *(required)* | Re-ranker Hub ID or local path |
| `--reranker-type` | `crossencoder` | `crossencoder` or `colbert` |
| `--dataset-version` | `v3` | SentencePairs version used as the test set |
| `--top-k` / `-k` | `5` | Candidates retrieved per query (use `50` for the paper) |
| `--redis-host` / `--redis-port` | `localhost` / `6379` | Redis connection |
| `--flush-cache` | `False` | Flush Redis before populating the cache |
| `--output` | auto-generated | Result JSON path |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

If `--output` is omitted, the file is named automatically:

```
eval_results_[biencoder_model_path=...]_[reranker_model_path=...]_[reranker_type=...]_[top_k=...].json
```

To sweep all combinations, use the shell helpers:

```bash
bash src/shell/run_reranker_evals.sh                                  # every retriever × re-ranker
bash src/shell/run_reranker_evals_for_retriever.sh <retriever> <redis_port> [top_k]   # one retriever, all re-rankers
```

Each result JSON records, per query: the retrieved candidates and scores, the re-ranked candidates and scores, the ground-truth label, and retrieval/reranking/total latencies — everything the analysis scripts need.

### 4. Analysis

Once `results/` is populated, the analysis scripts compute the paper's metrics and figures. All of them accept `--workers` (default `-1` = all CPUs).

#### Score Calibration

Fit **temperature** and **Platt** scaling parameters for a re-ranker on the train+val split. The output JSON is consumed by the classification and distribution analyses via `--calibration`. The paper applies and compares post-hoc calibration on both **BCE-** and **MNRL-trained** re-rankers; ColBERT-family models need no calibration, since their gap is entirely structural.

```bash
python src/analysis/compute_calibration.py \
  --model-path redis/langcache-reranker-v2-bce \
  --dataset-version v3 \
  --output calibration_params.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | *(required)* | Hub ID or local path of the re-ranker |
| `--dataset-version` | *(required)* | Version the model was trained on (`v1`/`v2`/`v3`) |
| `--output` | *(required)* | Calibration JSON to write/merge into |
| `--batch-size` | `64` | Inference batch size |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

To print the **calibration-parameter table** (Table: calib-params) — the fitted `T`, Platt `a`/`b`, and the change in exact P-CHR AUC under calibration — pass `--report` (no GPU). Δ P-CHR AUC is exactly `0.000` because temperature/Platt scaling are strictly monotone and P-CHR AUC is rank-invariant; supplying a native and a calibrated `cls_metrics.json` verifies this from data:

```bash
python src/analysis/compute_calibration.py --report --output calibration_params.json \
  --native-cls-metrics plots/classification/cls_metrics.json \
  --calibrated-cls-metrics plots/calibrated/cls_metrics.json
```

#### PR-AUC and P-CHR-AUC (classification)

Computes PR-AUC and **exact (grid-free)** Precision–CHR / Precision–VCHR AUC for every combination across `k = 1..K`, derives the operational-gap decomposition (`Δ_op`, `Δ_str`, `Δ_util`) and **ORR**, and renders the paper's curves. In one run over all combinations it prints, in addition to the per-combo tables, the **per-retriever baselines** table and the **per-reranker averages over retrievers** table (the two headline tables).

```bash
python src/analysis/analyze_cls.py \
  --results-dir results/ \
  --plots-dir plots/classification/ \
  --output plots/classification/cls_metrics.json \
  --calibration calibration_params.json \
  --calibration-method temperature
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | *(required)* | Directory of evaluation result JSONs |
| `--plots-dir` | *(required)* | Output directory for plots |
| `--output` | *(required)* | JSON summary of computed metrics (per-combo + `aggregates`) |
| `--transform` | `native` | Score transform for the normalization ablation: `native`, `pool_softmax`, or `raw_sigmoid` |
| `--thresholds` | `0.0–1.0` step `0.01` | Grid used only for the auxiliary F1-optimal precision/recall (the P-CHR/P-VCHR AUCs are grid-free) |
| `--calibration` | `None` | `calibration_params.json` from `compute_calibration.py` |
| `--calibration-method` | `temperature` | `temperature` or `platt` |
| `--workers` | `-1` | Parallel workers (`-1` = all CPUs) |

**Outputs:** `cls_metrics.json` (per-combo metrics + a top-level `aggregates` block with the retriever baselines and per-reranker averages), summary `*_vs_k.png` plots, and per-`k` `combined_pr_curves.png` / `combined_precision_chr_curves.png` / `combined_precision_vchr_curves.png`. The retriever baseline is drawn as a bold black dashed line, reranker-augmented systems as solid tab10 lines with canonical model names, sorted by AUC. Running with `--transform pool_softmax` / `raw_sigmoid` produces the score-normalization ablation.

#### Score Distribution Analysis

KDE plots of positive vs. negative ground-truth scores per retriever and per retriever–reranker pair, plus the full probability-calibration table (`Table: distribution-metrics`): ROC-AUC, KS statistic, KDE overlap, **ECE** (15 equal-width bins), **NLL** (BCE), **Brier**, and — when joined with the classification metrics — the exact **P-CHR AUC** per reranker. It also reports the pooled Spearman correlations of ECE / NLL / Brier against exact P-CHR over all combos (paper caption ρ ≈ 0.22 / 0.27 / 0.50).

```bash
# Run analyze_cls.py first so its cls_metrics.json exists for the P-CHR join.
python src/analysis/analyze_distribution.py \
  --results-dir results/ \
  --plots-dir plots/distribution/ \
  --output plots/distribution/dist_metrics.json \
  --calibration calibration_params.json \
  --cls-metrics cls_metrics.json
```

Arguments mirror `analyze_cls.py` (`--results-dir`, `--plots-dir`, `--output`, `--calibration`, `--calibration-method`, `--workers`), plus `--cls-metrics` — the `cls_metrics.json` written by `analyze_cls.py`; when given, each combo's exact P-CHR AUC is joined into the reranker table and the pooled ECE/NLL/Brier-vs-P-CHR Spearman correlations are printed. Outputs one KDE plot per unique retriever and per retriever–reranker pair (seaborn styling matching the camera-ready `fig:dist-kde`), a `rich` metrics table, and a metrics JSON.

#### Latency Analysis

Per-query retrieval / reranking / total latency (mean, std, p95) and reranking overhead, across all combinations (`Table: latency`).

```bash
python src/analysis/analyze_latency.py \
  --results-dir results/ \
  --plots-dir plots/latency/ \
  --output plots/latency/latency_metrics.json \
  --cls-metrics cls_metrics.json \
  --count-params
```

Arguments: `--results-dir`, `--plots-dir`, `--output` (all required), `--workers`, plus two optional joins used to reproduce the paper table:
- `--cls-metrics` — the `cls_metrics.json` from `analyze_cls.py`; adds the exact **P-CHR AUC** and **ΔRet** (reranker P-CHR minus the retriever's P-CHR) columns.
- `--count-params` — populates the **Size** column by loading each reranker on CPU and summing parameters. Off by default so the run needs no model downloads (the column shows `—` when omitted).

Outputs `latency_breakdown.png`, `latency_per_reranker.png`, `latency_per_retriever.png`, a `rich` table, and a metrics JSON.
