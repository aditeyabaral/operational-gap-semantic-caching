import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np
from rich.console import Console
from rich.table import Table


def _latest_params(model_entry: dict) -> dict:
    """Return the newest dataset-version's params from a compute_calibration model entry."""
    if not model_entry:
        return {}
    first_val = next(iter(model_entry.values()))
    if isinstance(first_val, dict) and "temperature" in first_val:
        return model_entry[sorted(model_entry.keys())[-1]]
    return model_entry


def _avg_reranker_pchr(cls_metrics_path: str) -> dict:
    """Map reranker (final path component) -> mean exact P-CHR AUC over retrievers, from a cls_metrics.json."""
    with open(cls_metrics_path) as f:
        data = json.load(f)
    by_reranker = {}
    for r in data.get("results", []):
        label = r["label"]
        reranker = label.split("+", 1)[1] if "+" in label else label
        kr = r["k_results"]
        k_max = max(kr, key=lambda k: int(k))
        by_reranker.setdefault(reranker, []).append(
            kr[k_max]["reranker_precision_chr_auc"]
        )
    return {rr: float(np.mean(v)) for rr, v in by_reranker.items()}


def report_calib_params_table(
    params_path: str,
    native_cls_metrics: str = None,
    calibrated_cls_metrics: str = None,
) -> None:
    """Print the post-hoc calibration parameter table (Table: calib-params) as a rich table.

    T, Platt a, and Platt b are read from the calibration params JSON produced by fitting.
    Delta P-CHR AUC (change in exact P-CHR AUC under temperature/Platt scaling, averaged over
    retrievers) is joined from a native vs calibrated cls_metrics.json pair when both are given;
    it is exactly 0.000 because temperature and Platt scaling are strictly monotone and exact
    P-CHR AUC is rank-invariant (Appendix: Post-Hoc Calibration).
    """
    with open(params_path) as f:
        params = json.load(f)

    native = _avg_reranker_pchr(native_cls_metrics) if native_cls_metrics else None
    calibrated = (
        _avg_reranker_pchr(calibrated_cls_metrics) if calibrated_cls_metrics else None
    )

    console = Console(width=200)
    table = Table(
        title="Post-hoc calibration parameters and their effect",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Reranker", justify="left")
    table.add_column("T", justify="right")
    table.add_column("Platt a", justify="right")
    table.add_column("Platt b", justify="right")
    table.add_column("Δ P-CHR AUC", justify="right")

    for model_key, model_entry in params.items():
        p = _latest_params(model_entry)
        if not p:
            continue
        if native is not None and calibrated is not None:
            d = calibrated.get(model_key, float("nan")) - native.get(
                model_key, float("nan")
            )
            delta = f"{d:.3f}"
        else:
            delta = "0.000"  # rank-invariance: strictly monotone calibration cannot move exact P-CHR
        table.add_row(
            model_key,
            f"{p['temperature']:.2f}",
            f"{p['platt_a']:.2f}",
            f"{p['platt_b']:.2f}",
            delta,
        )
    console.print(table)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit temperature T by minimizing NLL: calibrated_prob = sigmoid(logit / T)."""
    from scipy.optimize import minimize_scalar

    def nll(log_t):
        T = np.exp(log_t)
        probs = np.clip(1.0 / (1.0 + np.exp(-logits / T)), 1e-7, 1.0 - 1e-7)
        return -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling (a, b) via logistic regression: calibrated_prob = sigmoid(a * logit + b)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    clf.fit(logits.reshape(-1, 1), labels.astype(int))
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compute temperature and Platt scaling calibration parameters for a BCE reranker, "
        "or (with --report) print the calibration-parameter table."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Report mode: print the calibration-parameter table (Table: calib-params) from an "
        "existing --output params file (no GPU/fitting). Optionally join Δ P-CHR AUC from "
        "--native-cls-metrics and --calibrated-cls-metrics.",
    )
    parser.add_argument(
        "--native-cls-metrics",
        type=str,
        default=None,
        help="Report mode: cls_metrics.json from analyze_cls.py without calibration.",
    )
    parser.add_argument(
        "--calibrated-cls-metrics",
        type=str,
        default=None,
        help="Report mode: cls_metrics.json from analyze_cls.py with --calibration applied.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path of the BCE reranker model.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        choices=["v1", "v2", "v3"],
        help="Dataset version to use (must match the version the model was trained on).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSON file. If the file already exists, the new entry is merged in.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.report:
        report_calib_params_table(
            args.output, args.native_cls_metrics, args.calibrated_cls_metrics
        )
        sys.exit(0)

    if not args.model_path or not args.dataset_version:
        parser.error(
            "--model-path and --dataset-version are required unless --report is set."
        )

    print(args)

    import torch
    from sentence_transformers import CrossEncoder

    from src.reranker.util import load_langcache_sentencepairs_splits

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    # Load train+val using the same strategy as finetune_crossencoder.py
    print(
        f"Loading train+val from redis/langcache-sentencepairs-{args.dataset_version}..."
    )
    train_dataset, _, _ = load_langcache_sentencepairs_splits(
        subset_names={f"redis/langcache-sentencepairs-{args.dataset_version}": ["all"]},
        combine_train_and_val=True,
    )
    print(f"Loaded {len(train_dataset)} pairs.")

    pairs = list(zip(train_dataset["sentence1"], train_dataset["sentence2"]))
    labels = np.array(train_dataset["label"], dtype=np.float64)
    print(
        f"Label distribution: {int(labels.sum())} positives ({100 * labels.mean():.1f}%), "
        f"{int((1 - labels).sum())} negatives ({100 * (1 - labels.mean()):.1f}%)"
    )

    # Load model with identity activation to get raw logits consistent with eval pipeline
    print(f"Loading model: {args.model_path}")
    model = CrossEncoder(
        args.model_path,
        num_labels=1,
        activation_fn=torch.nn.Identity(),
        device=args.device,
        model_kwargs={"dtype": torch.bfloat16},
    )

    # Run inference
    print(f"Running inference on {len(pairs)} pairs (batch_size={args.batch_size})...")
    logits = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logits = np.array(logits, dtype=np.float64)
    print(
        f"Logit stats: min={logits.min():.4f}, max={logits.max():.4f}, "
        f"mean={logits.mean():.4f}, std={logits.std():.4f}"
    )

    # Fit temperature scaling
    print("Fitting temperature scaling...")
    T = fit_temperature(logits, labels)
    print(f"  T = {T:.6f}")

    # Fit Platt scaling
    print("Fitting Platt scaling...")
    a, b = fit_platt(logits, labels)
    print(f"  a = {a:.6f}, b = {b:.6f}")

    # Build output entry keyed by (model_name, dataset_version)
    model_key = args.model_path.rstrip("/").split("/")[-1]

    if os.path.exists(args.output):
        with open(args.output) as f:
            entry = json.load(f)
    else:
        entry = {}

    entry.setdefault(model_key, {})[args.dataset_version] = {
        "model_path": args.model_path,
        "num_samples": len(pairs),
        "temperature": T,
        "platt_a": a,
        "platt_b": b,
    }

    with open(args.output, "w") as f:
        json.dump(entry, f, indent=4)
    print(f"Saved calibration parameters to {args.output}")
