import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.table import Table
from scipy.stats import gaussian_kde, ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.analysis.analyze_cls import canonical_reranker, canonical_retriever
from src.analysis.util import (
    extract_models_from_filename,
    get_calibration_params,
    load_calibration,
    nan_to_none,
    normalize_reranker_scores,
)

# Camera-ready KDE styling (matches the paper's Score-Distribution figure).
sns.set_theme(style="darkgrid", context="talk", font_scale=1.0)
_POS_COLOR, _NEG_COLOR, _OVERLAP_COLOR = "#2ca02c", "#d62728", "#9467bd"

# Probability-calibration metrics (Appendix: Score Distribution Analysis). Computed on the
# ground-truth candidate's normalized score s(q,c*) against the binary label, exactly as the
# separation metrics are. These test whether probability calibration predicts deployment
# (P-CHR AUC); it does not (see the pooled Spearman correlations printed at the end).
_EPS = 1e-7


def ece_score(scores, labels, n_bins: int = 15) -> float:
    """Expected Calibration Error, 15 equal-width bins on [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(scores)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (
            (scores > lo) & (scores <= hi)
            if lo > 0
            else (scores >= lo) & (scores <= hi)
        )
        if not np.any(mask):
            continue
        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(labels[mask]))
        total += (np.sum(mask) / n) * abs(acc - conf)
    return float(total)


def nll_score(scores, labels) -> float:
    """Negative log-likelihood (binary cross-entropy) of the scores against the labels."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")
    p = np.clip(scores, _EPS, 1 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def brier_score(scores, labels) -> float:
    """Brier score: mean squared error between scores and labels."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def extract_gt_scores_labels(
    results,
    reranker_type=None,
    calib_params: dict = None,
    calibration_method: str = "temperature",
):
    """Extract GT retriever scores and normalized reranker scores with labels."""
    retriever_gt_scores = []
    reranker_gt_scores = []
    labels = []

    for example in results:
        gt = example["ground_truth"]
        label = example.get("label", 0)

        try:
            gt_idx = example["retrieved_candidates"].index(gt)
            retriever_score = example["retrieved_scores"][gt_idx]
        except ValueError:
            retriever_score = 0.0

        reranker_score_norm = 0.0

        ranked_scores = example.get("ranked_scores", [])
        ranked_candidates = example.get("ranked_candidates", [])

        try:
            gt_idx_rer = ranked_candidates.index(gt)
            normalized = normalize_reranker_scores(
                ranked_scores,
                reranker_type=reranker_type,
                calib_params=calib_params,
                calibration_method=calibration_method,
            )
            reranker_score_norm = float(normalized[gt_idx_rer])
        except Exception:
            reranker_score_norm = 0.0

        retriever_gt_scores.append(retriever_score)
        reranker_gt_scores.append(reranker_score_norm)
        labels.append(label)

    return retriever_gt_scores, reranker_gt_scores, labels


def ks_score(scores, labels):
    pos_scores = [s for s, lbl in zip(scores, labels) if lbl == 1]
    neg_scores = [s for s, lbl in zip(scores, labels) if lbl == 0]
    if not pos_scores or not neg_scores:
        return float("nan")
    stat, _ = ks_2samp(pos_scores, neg_scores)
    return stat


def area_of_overlap_kde(scores, labels, n_points=1000):
    pos_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 1])
    neg_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 0])
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    try:
        kde_pos = gaussian_kde(pos_scores)
        kde_neg = gaussian_kde(neg_scores)
    except np.linalg.LinAlgError:
        return float("nan")

    x_grid = np.linspace(0.0, 1.0, n_points)
    f_pos = kde_pos(x_grid)
    f_neg = kde_neg(x_grid)
    return float(np.trapezoid(np.minimum(f_pos, f_neg), x_grid))


def plot_distributions(scores, labels, title, output_path):
    pos_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 1])
    neg_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 0])

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        print(f"Cannot plot {title}: missing positive or negative samples")
        return

    x_grid = np.linspace(0.0, 1.0, 1000)
    try:
        kde_pos = gaussian_kde(pos_scores)
        kde_neg = gaussian_kde(neg_scores)
    except np.linalg.LinAlgError:
        print(
            f"Cannot plot {title}: degenerate score distribution (all scores identical)"
        )
        return

    f_pos = kde_pos(x_grid)
    f_neg = kde_neg(x_grid)
    overlap_density = np.minimum(f_pos, f_neg)
    overlap = np.trapezoid(overlap_density, x_grid)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_grid, f_pos, color=_POS_COLOR, linewidth=2.6, label="Positive", alpha=0.9)
    ax.plot(x_grid, f_neg, color=_NEG_COLOR, linewidth=2.6, label="Negative", alpha=0.9)
    ax.fill_between(
        x_grid, overlap_density, alpha=0.35, color=_OVERLAP_COLOR, label="Overlap"
    )

    stats_text = (
        f"Positive: μ={pos_scores.mean():.3f}, σ={pos_scores.std():.3f}\n"
        f"Negative: μ={neg_scores.mean():.3f}, σ={neg_scores.std():.3f}\n"
        f"Overlap: {overlap:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.65),
    )

    ax.set_xlabel("Score", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.tick_params(labelsize=15)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=15, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def process_file(args_tuple):
    filename, results_dir, calibration_data, calibration_method = args_tuple
    path = os.path.join(results_dir, filename)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return None
    results = data.get("results", [])
    if not results:
        return None

    retriever_name, reranker_name, reranker_type = extract_models_from_filename(
        filename
    )
    calib_params = get_calibration_params(reranker_name, calibration_data)
    retriever_scores, reranker_scores, labels = extract_gt_scores_labels(
        results,
        reranker_type=reranker_type,
        calib_params=calib_params,
        calibration_method=calibration_method,
    )

    try:
        retr_auc = roc_auc_score(labels, retriever_scores)
    except ValueError:
        retr_auc = float("nan")
    try:
        rerank_auc = roc_auc_score(labels, reranker_scores)
    except ValueError:
        rerank_auc = float("nan")

    retr_ks = ks_score(retriever_scores, labels)
    rer_ks = ks_score(reranker_scores, labels)
    retr_overlap_kde = area_of_overlap_kde(retriever_scores, labels)
    rer_overlap_kde = area_of_overlap_kde(reranker_scores, labels)

    # Probability-calibration metrics on the reranker's normalized GT score vs label.
    rer_ece = ece_score(reranker_scores, labels)
    rer_nll = nll_score(reranker_scores, labels)
    rer_brier = brier_score(reranker_scores, labels)

    return {
        "filename": filename,
        "retriever_name": retriever_name,
        "reranker_name": reranker_name,
        "retriever_scores": retriever_scores,
        "reranker_scores": reranker_scores,
        "labels": labels,
        "retr_auc": retr_auc,
        "rerank_auc": rerank_auc,
        "retr_ks": retr_ks,
        "rer_ks": rer_ks,
        "retr_overlap_kde": retr_overlap_kde,
        "rer_overlap_kde": rer_overlap_kde,
        "rer_ece": rer_ece,
        "rer_nll": rer_nll,
        "rer_brier": rer_brier,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot score distributions for results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--plots-dir", type=str, required=True, help="Directory to save output plots"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration_params.json produced by compute_calibration.py. Applied to any model whose key is found in the file.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["temperature", "platt"],
        default="temperature",
        help="Calibration method to apply when --calibration is provided (default: temperature).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save a JSON summary of computed metrics.",
    )
    parser.add_argument(
        "--cls-metrics",
        type=str,
        default=None,
        help=(
            "Path to the cls_metrics.json produced by analyze_cls.py. When given, each combo's "
            "exact P-CHR AUC is joined into the reranker table and the pooled Spearman "
            "correlations of ECE/NLL/Brier vs exact P-CHR (over all combos) are reported."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of parallel worker processes (-1 uses all available CPUs, default: -1)",
    )
    args = parser.parse_args()

    def _combo_key(retriever_name, reranker_name):
        """Join key shared with analyze_cls labels: final path components, '+'-separated."""
        if retriever_name is None or reranker_name is None:
            return None
        return f"{retriever_name.split('--')[-1]}+{reranker_name.split('--')[-1]}"

    # Optional: exact P-CHR AUC per combo from analyze_cls, keyed for the join above.
    pchr_by_combo = {}
    if args.cls_metrics:
        with open(args.cls_metrics) as f:
            cls_data = json.load(f)
        for r in cls_data.get("results", []):
            kr = r["k_results"]
            k_max = max(kr, key=lambda k: int(k))
            pchr_by_combo[r["label"]] = kr[k_max]["reranker_precision_chr_auc"]

    calibration_data = load_calibration(args.calibration)

    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in os.listdir(args.results_dir)
        if f.startswith("eval_results_") and f.endswith(".json")
    ]
    num_workers = (
        int(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) * 0.75)
        if args.workers == -1
        else args.workers
    )

    print("Configuration:")
    print(f"  Input directory:  {args.results_dir}")
    print(f"  Output directory: {args.plots_dir}")
    print(f"  Output JSON:      {args.output}")
    print(f"  Calibration:      {args.calibration or 'none'}")
    if args.calibration:
        print(f"  Calib method:     {args.calibration_method}")
    print(f"  Workers:          {num_workers}")

    worker_args = [
        (
            f,
            args.results_dir,
            calibration_data,
            args.calibration_method,
        )
        for f in files
    ]

    print(f"\nProcessing {len(files)} files with {num_workers} worker(s)...")
    with Pool(processes=num_workers) as pool:
        file_results = [
            r
            for r in tqdm(
                pool.imap(process_file, worker_args),
                total=len(worker_args),
                desc="Files completed",
                dynamic_ncols=True,
            )
            if r is not None
        ]

    # Plotting and deduplication are sequential (matplotlib is not parallel-safe)
    saved_retrievers = set()
    saved_pairs = set()
    retriever_metrics = []  # one entry per unique retriever
    reranker_metrics = []  # one entry per retriever+reranker pair

    for result in file_results:
        filename = result["filename"]
        retriever_name = result["retriever_name"]
        reranker_name = result["reranker_name"]
        retriever_scores = result["retriever_scores"]
        reranker_scores = result["reranker_scores"]
        labels = result["labels"]

        base_name = filename.replace(".json", "")

        # Retriever plot: one KDE per unique retriever
        if retriever_name is None:
            plot_distributions(
                retriever_scores,
                labels,
                title=f"{base_name} Retriever Score Distribution",
                output_path=os.path.join(
                    args.plots_dir, f"{base_name}_retriever_kde.png"
                ),
            )
            retriever_metrics.append(
                {
                    "label": base_name,
                    "retr_auc": result["retr_auc"],
                    "retr_ks": result["retr_ks"],
                    "retr_overlap_kde": result["retr_overlap_kde"],
                }
            )
        elif retriever_name not in saved_retrievers:
            plot_distributions(
                retriever_scores,
                labels,
                title=f"{canonical_retriever(retriever_name.split('--')[-1])} (retriever)",
                output_path=os.path.join(
                    args.plots_dir, f"{retriever_name}_retriever_kde.png"
                ),
            )
            retriever_metrics.append(
                {
                    "label": retriever_name,
                    "retr_auc": result["retr_auc"],
                    "retr_ks": result["retr_ks"],
                    "retr_overlap_kde": result["retr_overlap_kde"],
                }
            )
            saved_retrievers.add(retriever_name)

        # Reranker plot: one KDE per unique retriever+reranker pair
        if retriever_name is None or reranker_name is None:
            plot_distributions(
                reranker_scores,
                labels,
                title=f"{base_name} Reranker Score Distribution",
                output_path=os.path.join(
                    args.plots_dir, f"{base_name}_reranker_kde.png"
                ),
            )
            reranker_metrics.append(
                {
                    "label": base_name,
                    "retriever_name": retriever_name,
                    "reranker_name": reranker_name,
                    "rerank_auc": result["rerank_auc"],
                    "rer_ks": result["rer_ks"],
                    "rer_overlap_kde": result["rer_overlap_kde"],
                    "rer_ece": result["rer_ece"],
                    "rer_nll": result["rer_nll"],
                    "rer_brier": result["rer_brier"],
                }
            )
        else:
            pair = (retriever_name, reranker_name)
            if pair not in saved_pairs:
                plot_distributions(
                    reranker_scores,
                    labels,
                    title=canonical_reranker(reranker_name.split("--")[-1]),
                    output_path=os.path.join(
                        args.plots_dir,
                        f"{retriever_name}__{reranker_name}_reranker_kde.png",
                    ),
                )
                reranker_metrics.append(
                    {
                        "label": f"{retriever_name}+{reranker_name}",
                        "retriever_name": retriever_name,
                        "reranker_name": reranker_name,
                        "rerank_auc": result["rerank_auc"],
                        "rer_ks": result["rer_ks"],
                        "rer_overlap_kde": result["rer_overlap_kde"],
                        "rer_ece": result["rer_ece"],
                        "rer_nll": result["rer_nll"],
                        "rer_brier": result["rer_brier"],
                    }
                )
                saved_pairs.add(pair)

    console = Console(width=300)

    ret_table = Table(
        title="Retriever Score Distribution Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    ret_table.add_column("Retriever", style="dim", no_wrap=True)
    ret_table.add_column("ROC AUC", justify="right", min_width=8)
    ret_table.add_column("KS Stat", justify="right", min_width=7)
    ret_table.add_column("KDE Overlap", justify="right", min_width=11)
    for entry in retriever_metrics:
        ret_table.add_row(
            entry["label"],
            f"{entry['retr_auc']:.4f}",
            f"{entry['retr_ks']:.4f}",
            f"{entry['retr_overlap_kde']:.4f}",
        )
    console.print(ret_table)

    # Join exact P-CHR AUC per combo (if provided) for the table and the correlations.
    for entry in reranker_metrics:
        key = _combo_key(entry.get("retriever_name"), entry.get("reranker_name"))
        entry["p_chr_auc"] = pchr_by_combo.get(key) if key is not None else None

    rer_table = Table(
        title="Reranker Score Distribution & Probability-Calibration Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    rer_table.add_column("Setup", style="dim", no_wrap=True)
    rer_table.add_column("ROC AUC", justify="right", min_width=8)
    rer_table.add_column("KS Stat", justify="right", min_width=7)
    rer_table.add_column("KDE Overlap", justify="right", min_width=11)
    rer_table.add_column("ECE", justify="right", min_width=6)
    rer_table.add_column("NLL", justify="right", min_width=6)
    rer_table.add_column("Brier", justify="right", min_width=6)
    rer_table.add_column("P-CHR AUC", justify="right", min_width=9)
    for entry in reranker_metrics:
        pchr = entry.get("p_chr_auc")
        rer_table.add_row(
            entry["label"],
            f"{entry['rerank_auc']:.4f}",
            f"{entry['rer_ks']:.4f}",
            f"{entry['rer_overlap_kde']:.4f}",
            f"{entry['rer_ece']:.3f}",
            f"{entry['rer_nll']:.2f}",
            f"{entry['rer_brier']:.3f}",
            "—" if pchr is None else f"{pchr:.3f}",
        )
    console.print(rer_table)

    # Pooled Spearman correlations of probability-calibration metrics vs exact P-CHR AUC.
    correlations = {}
    paired = [
        (e["rer_ece"], e["rer_nll"], e["rer_brier"], e["p_chr_auc"])
        for e in reranker_metrics
        if e.get("p_chr_auc") is not None
        and all(
            not (isinstance(v, float) and np.isnan(v))
            for v in (e["rer_ece"], e["rer_nll"], e["rer_brier"])
        )
    ]
    if len(paired) >= 3:
        ece_v, nll_v, brier_v, pchr_v = (np.array(c, dtype=float) for c in zip(*paired))
        corr_table = Table(
            title=f"Probability-calibration metrics vs exact P-CHR AUC (Spearman, n={len(paired)})",
            show_header=True,
            header_style="bold magenta",
        )
        corr_table.add_column("Metric", justify="left")
        corr_table.add_column("Spearman ρ", justify="right")
        corr_table.add_column("p-value", justify="right")
        for name, vals in (("ECE", ece_v), ("NLL", nll_v), ("Brier", brier_v)):
            rho, pval = spearmanr(vals, pchr_v)
            correlations[name.lower()] = {
                "spearman_vs_p_chr_auc": float(rho),
                "pval": float(pval),
            }
            corr_table.add_row(name, f"{rho:.3f}", f"{pval:.2e}")
        console.print(corr_table)
    elif args.cls_metrics:
        print("Not enough combos with a P-CHR join to compute correlations (need ≥ 3).")

    output_data = {
        "calibration": args.calibration,
        "calibration_method": args.calibration_method if args.calibration else None,
        "retriever_results": [
            {
                "label": entry["label"],
                "retr_auc": nan_to_none(entry["retr_auc"]),
                "retr_ks": nan_to_none(entry["retr_ks"]),
                "retr_overlap_kde": nan_to_none(entry["retr_overlap_kde"]),
            }
            for entry in retriever_metrics
        ],
        "reranker_results": [
            {
                "label": entry["label"],
                "rerank_auc": nan_to_none(entry["rerank_auc"]),
                "rer_ks": nan_to_none(entry["rer_ks"]),
                "rer_overlap_kde": nan_to_none(entry["rer_overlap_kde"]),
                "rer_ece": nan_to_none(entry["rer_ece"]),
                "rer_nll": nan_to_none(entry["rer_nll"]),
                "rer_brier": nan_to_none(entry["rer_brier"]),
                "p_chr_auc": nan_to_none(entry.get("p_chr_auc"))
                if entry.get("p_chr_auc") is not None
                else None,
            }
            for entry in reranker_metrics
        ],
        "calibration_vs_pchr_spearman": correlations,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved metrics to {args.output}")
