# https://sbert.net/docs/sentence_transformer/training_overview.html

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import multiprocessing
import random

import numpy as np
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.training_args import BatchSamplers

from src.reranker.util import (
    load_langcache_sentencepairs_splits,
    to_infonce,
)
from src.retriever.cache_evaluator import BiEncoderCacheEvaluator
from src.retriever.losses import ArcFaceInBatchLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine-tune a BiEncoder model")
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Path to the pre-trained biencoder model.",
    )
    parser.add_argument(
        "--finetuned-model-path",
        type=str,
        default="redis/langcache-embed-v3-small",
        help="Path to the finetuned biencoder model.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v2",
        help="Version of the LangCache Sentence Pairs dataset to use.",
    )
    parser.add_argument(
        "--train-dataset-subsets",
        default=["all"],
        nargs="+",
        help="Subset names of the dataset to use for training. If not provided, 'all' subset will be used.",
    )
    parser.add_argument(
        "--combine-train-and-val",
        action="store_true",
        help="Combine train and val splits into a single train split.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to use for evaluation during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=1_000,
        help="Number of steps to evaluate the model.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=10_000,
        help="Number of steps to save the model.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1_000,
        help="Number of steps to log the metrics.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to save.",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=1,
        help="Number of hard negatives to sample for each anchor, from its labelled negative pairs.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=20.0,
        help="Scale (inverse temperature) for the ArcFace loss.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Additive angular margin in radians for the ArcFace loss. 0 reduces to MNRL.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.10,
        help="Warmup as a fraction of total training steps.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.003, help="Weight decay."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum input sequence length in tokens; longer inputs are truncated.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/dlami/nvme/langcache-embed-models",
        help="Directory to save checkpoints and final model. "
        "A subdirectory will be created for the weights and metrics.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb name to use for logging.",
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
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of processes to use for dataset processing.",
    )
    args = parser.parse_args()
    print(args)

    # set seed for reproducibility
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = args.device

    # create output directory if it doesn't exist
    save_dir = f"{args.output_dir}/{args.finetuned_model_path.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)

    # load train and val datasets
    train_dataset, val_dataset, test_dataset = load_langcache_sentencepairs_splits(
        subset_names={
            f"redis/langcache-sentencepairs-{args.dataset_version}": args.train_dataset_subsets
        },
        combine_train_and_val=args.combine_train_and_val,
    )
    # transform to (anchor, positive, negative_1, ...) InfoNCE format for the in-batch loss
    train_dataset_infonce = to_infonce(
        train_dataset, num_negatives=args.num_negatives, seed=args.seed
    )
    val_dataset_infonce, test_dataset_infonce = None, None
    if val_dataset is not None:
        val_dataset_infonce = to_infonce(
            val_dataset, num_negatives=args.num_negatives, seed=args.seed + 1
        )
    if test_dataset is not None:
        test_dataset_infonce = to_infonce(
            test_dataset, num_negatives=args.num_negatives, seed=args.seed + 2
        )

    print(
        f"Loaded {len(train_dataset)} train samples, {len(val_dataset) if val_dataset is not None else 0} val samples, {len(test_dataset) if test_dataset is not None else 0} test samples"
    )
    print(f"Loaded {len(train_dataset_infonce)} train samples in InfoNCE format.")
    if val_dataset_infonce is not None:
        print(f"Loaded {len(val_dataset_infonce)} val samples in InfoNCE format.")
    if test_dataset_infonce is not None:
        print(f"Loaded {len(test_dataset_infonce)} test samples in InfoNCE format.")

    train_dataset_entries = [
        {
            "name": f"LangCache Sentence Pairs (subsets={args.train_dataset_subsets}, train+val={args.combine_train_and_val})",
            "id": f"redis/langcache-sentencepairs-{args.dataset_version}",
        }
    ]
    eval_dataset_entries = [
        {
            "name": f"LangCache Sentence Pairs (split={args.eval_split})",
            "id": f"redis/langcache-sentencepairs-{args.dataset_version}",
        }
    ]

    # create evaluators for validation
    if val_dataset is not None:
        val_evaluators_list = []
        val_evaluators_list.append(
            BinaryClassificationEvaluator(
                sentences1=val_dataset["sentence1"],
                sentences2=val_dataset["sentence2"],
                labels=val_dataset["label"],
                batch_size=args.batch_size,
                show_progress_bar=False,
                similarity_fn_names=["cosine"],
                name="val_cls",
            )
        )
        val_evaluators_list.append(
            BiEncoderCacheEvaluator(
                sentence_pairs=list(
                    zip(val_dataset["sentence1"], val_dataset["sentence2"])
                ),
                labels=val_dataset["label"],
                batch_size=args.batch_size,
                name="val_cache",
                device=device,
            )
        )
        val_evaluator = SequentialEvaluator(val_evaluators_list)
    else:
        val_evaluator = None

    # create evaluators for test
    if test_dataset is not None:
        test_evaluators_list = []
        test_evaluators_list.append(
            BinaryClassificationEvaluator(
                sentences1=test_dataset["sentence1"],
                sentences2=test_dataset["sentence2"],
                labels=test_dataset["label"],
                batch_size=args.batch_size,
                show_progress_bar=False,
                similarity_fn_names=["cosine"],
                name="test_cls",
            )
        )
        test_evaluators_list.append(
            BiEncoderCacheEvaluator(
                sentence_pairs=list(
                    zip(test_dataset["sentence1"], test_dataset["sentence2"])
                ),
                labels=test_dataset["label"],
                batch_size=args.batch_size,
                name="test_cache",
                device=device,
            )
        )
        test_evaluator = SequentialEvaluator(test_evaluators_list)
    else:
        test_evaluator = None

    # create model card data
    card_model_name = (
        "Redis fine-tuned BiEncoder model for semantic caching on LangCache"
    )
    card_model_id = args.finetuned_model_path
    model_card_data = SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name=card_model_name,
        model_id=card_model_id,
        task_name="sentence pair similarity",
        train_datasets=train_dataset_entries,
        eval_datasets=eval_dataset_entries,
        tags=[
            "biencoder",
            "sentence-transformers",
            "text-classification",
            "sentence-pair-classification",
            "semantic-similarity",
            "semantic-search",
            "retrieval",
            "reranking",
        ],
    )

    # load model
    model = SentenceTransformer(
        args.pretrained_model_path,
        model_card_data=model_card_data,
        device=device,
        model_kwargs={
            "attn_implementation": "sdpa",
            # drop the unused BERT pooler so DDP sees no parameters without gradients
            "add_pooling_layer": False,
            "dtype": torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32,
        },
    )
    model.max_seq_length = args.max_length

    # initialize loss
    loss = ArcFaceInBatchLoss(model, scale=args.scale, margin=args.margin)

    # initialize training args
    training_args = SentenceTransformerTrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        weight_decay=args.weight_decay,
        eval_on_start=True,
        eval_strategy="steps",
        save_strategy="steps",
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=f"{save_dir}/logs",
        report_to="wandb",
        run_name=args.wandb_run_name
        if args.wandb_run_name is not None
        else args.finetuned_model_path.split("/")[-1],
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.eval_split}_cls_cosine_f1",
        push_to_hub=True,
        hub_model_id=card_model_id,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    # initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_infonce,
        eval_dataset=test_dataset_infonce
        if args.eval_split == "test"
        else val_dataset_infonce,
        evaluator=test_evaluator if args.eval_split == "test" else val_evaluator,
        loss=loss,
    )

    # push the initial model to the hub
    model.push_to_hub(card_model_id, exist_ok=True)

    # train model
    trainer.train()

    # save model to HuggingFace Hub
    model.push_to_hub(card_model_id, exist_ok=True)

    # evaluate final model
    if val_evaluator is not None:
        val_scores = val_evaluator(model=model)
    else:
        val_scores = {}
    if test_evaluator is not None:
        test_scores = test_evaluator(model=model)
    else:
        test_scores = {}
    scores = {**val_scores, **test_scores}
    print(f"Final model scores: {scores}")
    with open(f"{save_dir}/final_metrics.json", "w") as f:
        json.dump(scores, f, indent=4)
