import argparse
import sys

from datasets import load_dataset

sys.path.insert(0, ".")

import json
import random

import numpy as np
import torch

from src.eval.retrieve_rerank_evaluator import RetrieveAndRerankEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained BiEncoder and Reranker model pipeline"
    )
    parser.add_argument(
        "--biencoder-model-path",
        type=str,
        required=True,
        help="Path to the pretrained biencoder model.",
    )
    parser.add_argument(
        "--reranker-model-path",
        type=str,
        required=True,
        help="Path to the pretrained reranker model.",
    )
    parser.add_argument(
        "--reranker-type",
        type=str,
        default="crossencoder",
        choices=["crossencoder", "colbert"],
        help="Type of re-ranker to use.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v3",
        help="Version of the Sentence Pairs dataset to use.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Top k results to retrieve from the cache.",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port.",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="Flush the cache before running the evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the evaluation results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the evaluation on (e.g., 'cuda', 'mps' or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    print(args)

    # set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = args.device

    # dataset contains sentence1, sentence2 and label
    dataset = load_dataset(
        f"redis/langcache-sentencepairs-{args.dataset_version}", "all", split="test"
    )
    # create the cache
    cache_sentences = list(set(dataset["sentence2"]))
    print(f"Found {len(cache_sentences)} unique sentences for cache.")

    # initialize the evaluator
    evaluator = RetrieveAndRerankEvaluator(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        biencoder_model_path=args.biencoder_model_path,
        reranker_model_path=args.reranker_model_path,
        reranker_type=args.reranker_type,
        top_k=args.top_k,
        device=device,
        flush_cache=args.flush_cache,
    )

    # populate cache
    evaluator.populate_cache(cache_sentences)
    cache_size = evaluator.get_cache_size()
    assert cache_size == len(cache_sentences), (
        f"Cache population failed due to size mismatch: found {cache_size} items in cache, expected {len(cache_sentences)}."
    )
    print("Cache population complete. Cache size:", cache_size)

    # prepare data for evaluation
    queries = [row["sentence1"] for row in dataset]
    ground_truths = [row["sentence2"] for row in dataset]
    labels = [row["label"] for row in dataset]

    # run evaluation
    eval_results = evaluator.evaluate(queries, ground_truths, labels)
    print("Evaluation complete.")

    # enrich results with dataset-specific metadata
    for i, result in enumerate(eval_results["results"]):
        row = dataset[i]
        result["source"] = row["source"]
        result["source_idx"] = row["source_idx"]

    # save results
    if args.output is None:
        sanitized_biencoder_model_path = args.biencoder_model_path.replace("/", "--")
        sanitized_reranker_model_path = args.reranker_model_path.replace("/", "--")
        args.output = (
            "eval_results"
            + f"_[biencoder_model_path={sanitized_biencoder_model_path}]"
            + f"_[reranker_model_path={sanitized_reranker_model_path}]"
            + f"_[reranker_type={args.reranker_type}]"
            + f"_[top_k={args.top_k}].json"
        )

    with open(args.output, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saved results to {args.output}")
