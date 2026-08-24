import argparse
import json
import os
from multiprocessing import Pool
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from src.analysis.util import extract_models_from_filename, nan_to_none

_RETRIEVAL_COLOR = "#4878CF"
_OVERHEAD_COLOR = "#D65F0A"
_ERRORBAR_COLOR = "#333333"


def _fmt_size(n_params: int | None) -> str:
    """Human-readable parameter count, e.g. 109482240 -> '110M'."""
    if n_params is None:
        return "—"
    if n_params >= 1e9:
        return f"{n_params / 1e9:.1f}B"
    return f"{round(n_params / 1e6)}M"


def count_model_params(sanitized_name: str) -> int | None:
    """Total parameter count for a reranker, loaded from its weights (paper Size column).

    ``sanitized_name`` is the filename form ('colbert-ir--colbertv2.0'); the HF id is recovered by
    turning '--' back into '/'. The model is loaded on CPU purely to count parameters. Returns None
    if it cannot be loaded (e.g. offline), so the Size column degrades to '—' rather than crashing.
    """
    if not sanitized_name:
        return None
    hf_id = sanitized_name.replace("--", "/", 1)
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        n = int(sum(p.numel() for p in model.parameters()))
        del model
        return n
    except Exception as e:
        print(f"  [size] could not load {hf_id}: {type(e).__name__}: {e}")
        return None


def extract_durations(
    results: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ret, rer, tot = [], [], []
    for item in results:
        r = item.get("retrieval_duration")
        k = item.get("reranking_duration")
        t = item.get("total_duration")
        if r is None or k is None or t is None:
            continue
        try:
            r, k, t = float(r), float(k), float(t)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(r) and np.isfinite(k) and np.isfinite(t)):
            continue
        ret.append(r)
        rer.append(k)
        tot.append(t)
    return (
        np.array(ret, dtype=np.float64),
        np.array(rer, dtype=np.float64),
        np.array(tot, dtype=np.float64),
    )


def compute_latency_stats(
    retrieval_ms: np.ndarray,
    reranking_ms: np.ndarray,
    total_ms: np.ndarray,
) -> dict[str, float]:
    n = len(total_ms)
    ddof = 1 if n > 1 else 0
    return {
        "mean_retrieval": float(np.mean(retrieval_ms)),
        "std_retrieval": float(np.std(retrieval_ms, ddof=ddof)),
        "p95_retrieval": float(np.percentile(retrieval_ms, 95)),
        "mean_reranking": float(np.mean(reranking_ms)),
        "std_reranking": float(np.std(reranking_ms, ddof=ddof)),
        "p95_reranking": float(np.percentile(reranking_ms, 95)),
        "mean_total": float(np.mean(total_ms)),
        "std_total": float(np.std(total_ms, ddof=ddof)),
        "p95_total": float(np.percentile(total_ms, 95)),
        "n": n,
    }


def _truncate_label(label: str, maxlen: int = 75) -> str:
    if len(label) <= maxlen:
        return label
    keep = (maxlen - 3) // 2
    return label[:keep] + "…" + label[-(maxlen - keep - 1) :]


def process_file(
    args_tuple: tuple[str, str],
) -> dict[str, Any] | None:
    filename, results_dir = args_tuple
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
    retriever_short = retriever_name.split("--")[-1] if retriever_name else "unknown"
    reranker_short = reranker_name.split("--")[-1] if reranker_name else "unknown"
    label = f"{retriever_short}+{reranker_short}"

    retrieval_ms, reranking_ms, total_ms = extract_durations(results)
    if len(total_ms) == 0:
        return None

    stats = compute_latency_stats(retrieval_ms, reranking_ms, total_ms)
    return {
        "label": label,
        "retriever_name": retriever_name,
        "reranker_name": reranker_name,
        "reranker_type": reranker_type,
        **stats,
    }


def _plot_vertical_bars(
    segments: list[tuple[np.ndarray, str, str]],
    labels: list[str],
    title: str,
    output_path: str,
) -> None:
    N = len(labels)
    x_pos = np.arange(N)

    _fig, ax = plt.subplots(figsize=(max(14, 0.5 * N + 3), 8))
    bottom = np.zeros(N)
    for values, color, legend_label in segments:
        ax.bar(
            x_pos, values, bottom=bottom, color=color, width=0.65, label=legend_label
        )
        bottom += values

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean latency per query (ms)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.grid(True, color="#E0E0E0", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_latency_bars(
    stats: list[dict[str, Any]],
    output_path: str,
) -> None:
    if not stats:
        print("No data to plot.")
        return

    stats_sorted = sorted(stats, key=lambda x: x["mean_total"])
    mean_ret = np.array([s["mean_retrieval"] for s in stats_sorted])
    mean_over = np.array([s["mean_total"] - s["mean_retrieval"] for s in stats_sorted])
    _plot_vertical_bars(
        segments=[
            (mean_ret, _RETRIEVAL_COLOR, "Retrieval latency"),
            (mean_over, _OVERHEAD_COLOR, "Reranking overhead"),
        ],
        labels=[_truncate_label(s["label"]) for s in stats_sorted],
        title="Retrieval + Reranking Latency Breakdown per Model Combination",
        output_path=output_path,
    )


def plot_latency_per_reranker(
    stats: list[dict[str, Any]],
    output_path: str,
) -> None:
    if not stats:
        print("No data to plot.")
        return

    from collections import defaultdict

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in stats:
        key = s["reranker_name"].split("--")[-1] if s["reranker_name"] else "unknown"
        groups[key].append(s)

    reranker_stats = sorted(
        [
            {
                "label": reranker,
                "mean_reranking": float(
                    np.mean([e["mean_reranking"] for e in entries])
                ),
            }
            for reranker, entries in groups.items()
        ],
        key=lambda x: x["mean_reranking"],
    )

    _plot_vertical_bars(
        segments=[
            (
                np.array([s["mean_reranking"] for s in reranker_stats]),
                _OVERHEAD_COLOR,
                "Avg reranking latency",
            )
        ],
        labels=[_truncate_label(s["label"]) for s in reranker_stats],
        title="Average Reranking Latency per Reranker",
        output_path=output_path,
    )


def plot_latency_per_retriever(
    stats: list[dict[str, Any]],
    output_path: str,
) -> None:
    if not stats:
        print("No data to plot.")
        return

    from collections import defaultdict

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in stats:
        key = s["retriever_name"].split("--")[-1] if s["retriever_name"] else "unknown"
        groups[key].append(s)

    retriever_stats = sorted(
        [
            {
                "label": retriever,
                "mean_retrieval": float(
                    np.mean([e["mean_retrieval"] for e in entries])
                ),
            }
            for retriever, entries in groups.items()
        ],
        key=lambda x: x["mean_retrieval"],
    )

    _plot_vertical_bars(
        segments=[
            (
                np.array([s["mean_retrieval"] for s in retriever_stats]),
                _RETRIEVAL_COLOR,
                "Avg retrieval latency",
            )
        ],
        labels=[_truncate_label(s["label"]) for s in retriever_stats],
        title="Average Retrieval Latency per Retriever",
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze per-query retrieval and reranking latency per model combination."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation result JSON files",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        required=True,
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save JSON latency summary",
    )
    parser.add_argument(
        "--cls-metrics",
        type=str,
        default=None,
        help=(
            "Path to the cls_metrics.json produced by analyze_cls.py. When given, exact P-CHR AUC "
            "and ΔRet (reranker P-CHR minus the retriever's P-CHR) are joined into the per-combo "
            "latency table (Table: latency)."
        ),
    )
    parser.add_argument(
        "--count-params",
        action="store_true",
        help="Populate the Size column by loading each reranker on CPU and counting parameters "
        "(paper Table: latency). Off by default so the run needs no model downloads.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of parallel worker processes (-1 uses all available CPUs, default: -1)",
    )
    args = parser.parse_args()

    def _combo_key(retriever_name, reranker_name):
        if retriever_name is None or reranker_name is None:
            return None
        return f"{retriever_name.split('--')[-1]}+{reranker_name.split('--')[-1]}"

    # Optional join: exact P-CHR AUC (reranker) and the retriever baseline P-CHR per combo.
    pchr_by_combo, retr_pchr_by_combo = {}, {}
    if args.cls_metrics:
        with open(args.cls_metrics) as f:
            cls_data = json.load(f)
        for r in cls_data.get("results", []):
            kr = r["k_results"]
            k_max = max(kr, key=lambda k: int(k))
            pchr_by_combo[r["label"]] = kr[k_max]["reranker_precision_chr_auc"]
            retr_pchr_by_combo[r["label"]] = kr[k_max]["retriever_precision_chr_auc"]

    num_workers = (
        int(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) * 0.75)
        if args.workers == -1
        else args.workers
    )

    print("Configuration:")
    print(f"  Input directory:  {args.results_dir}")
    print(f"  Output directory: {args.plots_dir}")
    print(f"  Output JSON:      {args.output}")
    print(f"  Workers:          {num_workers}")

    files = [
        f
        for f in os.listdir(args.results_dir)
        if f.endswith(".json") and f.startswith("eval_results_")
    ]
    worker_args = [(f, args.results_dir) for f in files]

    print(f"\nProcessing {len(files)} files with {num_workers} worker(s)...")
    with Pool(processes=num_workers) as pool:
        all_stats = [
            r
            for r in tqdm(
                pool.imap(process_file, worker_args),
                total=len(worker_args),
                desc="Files completed",
                leave=True,
                dynamic_ncols=True,
            )
            if r is not None
        ]

    all_stats_sorted = sorted(all_stats, key=lambda x: x["mean_total"], reverse=True)

    # Optional: parameter count per unique reranker (paper Size column), loaded from weights once.
    size_by_reranker = {}
    if args.count_params:
        print("\nCounting model parameters (Size column)...")
        for name in sorted(
            {s["reranker_name"] for s in all_stats if s["reranker_name"]}
        ):
            size_by_reranker[name] = count_model_params(name)

    console = Console(width=300)
    table = Table(
        title="Latency Statistics per Model Combination",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Setup", style="dim", no_wrap=True)
    table.add_column("N", justify="right", min_width=8)
    table.add_column("Size", justify="right", min_width=6)
    table.add_column("P-CHR AUC", justify="right", min_width=9)
    table.add_column("ΔRet", justify="right", min_width=7)
    table.add_column("Avg Reranking (ms)", justify="right", min_width=20)
    table.add_column("p95 Reranking (ms)", justify="right", min_width=20)
    table.add_column("Avg Total (ms)", justify="right", min_width=18)
    table.add_column("Reranking Overhead %", justify="right", min_width=20)

    for s in all_stats_sorted:
        overhead_pct = (
            100.0 * (s["mean_total"] - s["mean_retrieval"]) / s["mean_total"]
            if s["mean_total"] > 0
            else 0.0
        )
        key = _combo_key(s.get("retriever_name"), s.get("reranker_name"))
        pchr = pchr_by_combo.get(key) if key is not None else None
        retr_pchr = retr_pchr_by_combo.get(key) if key is not None else None
        s["p_chr_auc"] = pchr
        s["delta_ret"] = (
            (pchr - retr_pchr) if (pchr is not None and retr_pchr is not None) else None
        )
        s["n_params"] = size_by_reranker.get(s.get("reranker_name"))
        table.add_row(
            s["label"],
            str(s["n"]),
            _fmt_size(s["n_params"]),
            "—" if pchr is None else f"{pchr:.3f}",
            "—" if s["delta_ret"] is None else f"{s['delta_ret']:+.3f}",
            f"{s['mean_reranking']:.4f} ± {s['std_reranking']:.4f}",
            f"{s['p95_reranking']:.4f}",
            f"{s['mean_total']:.4f} ± {s['std_total']:.4f}",
            f"{overhead_pct:.1f}%",
        )
    console.print(table)

    from collections import defaultdict

    reranker_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in all_stats:
        key = s["reranker_name"].split("--")[-1] if s["reranker_name"] else "unknown"
        reranker_groups[key].append(s)
    reranker_table_rows = sorted(
        [
            {
                "reranker": k,
                "mean_reranking": float(np.mean([e["mean_reranking"] for e in v])),
                "std_reranking": float(
                    np.std(
                        [e["mean_reranking"] for e in v], ddof=1 if len(v) > 1 else 0
                    )
                ),
                "n_combos": len(v),
            }
            for k, v in reranker_groups.items()
        ],
        key=lambda x: x["mean_reranking"],
    )
    reranker_table = Table(
        title="Average Reranking Latency per Reranker",
        show_header=True,
        header_style="bold cyan",
    )
    reranker_table.add_column("Reranker", style="dim", no_wrap=True)
    reranker_table.add_column("Combos", justify="right", min_width=8)
    reranker_table.add_column("Avg Reranking (ms)", justify="right", min_width=22)
    for r in reranker_table_rows:
        reranker_table.add_row(
            r["reranker"],
            str(r["n_combos"]),
            f"{r['mean_reranking']:.4f} ± {r['std_reranking']:.4f}",
        )
    console.print(reranker_table)

    retriever_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in all_stats:
        key = s["retriever_name"].split("--")[-1] if s["retriever_name"] else "unknown"
        retriever_groups[key].append(s)
    retriever_table_rows = sorted(
        [
            {
                "retriever": k,
                "mean_retrieval": float(np.mean([e["mean_retrieval"] for e in v])),
                "std_retrieval": float(
                    np.std(
                        [e["mean_retrieval"] for e in v], ddof=1 if len(v) > 1 else 0
                    )
                ),
                "n_combos": len(v),
            }
            for k, v in retriever_groups.items()
        ],
        key=lambda x: x["mean_retrieval"],
    )
    retriever_table = Table(
        title="Average Retrieval Latency per Retriever",
        show_header=True,
        header_style="bold cyan",
    )
    retriever_table.add_column("Retriever", style="dim", no_wrap=True)
    retriever_table.add_column("Combos", justify="right", min_width=8)
    retriever_table.add_column("Avg Retrieval (ms)", justify="right", min_width=22)
    for r in retriever_table_rows:
        retriever_table.add_row(
            r["retriever"],
            str(r["n_combos"]),
            f"{r['mean_retrieval']:.4f} ± {r['std_retrieval']:.4f}",
        )
    console.print(retriever_table)

    output_data = {
        "results_dir": args.results_dir,
        "results": [
            {
                "label": s["label"],
                "retriever_name": s["retriever_name"],
                "reranker_name": s["reranker_name"],
                "reranker_type": s["reranker_type"],
                "n": s["n"],
                "mean_retrieval_ms": nan_to_none(s["mean_retrieval"]),
                "std_retrieval_ms": nan_to_none(s["std_retrieval"]),
                "p95_retrieval_ms": nan_to_none(s["p95_retrieval"]),
                "mean_reranking_ms": nan_to_none(s["mean_reranking"]),
                "std_reranking_ms": nan_to_none(s["std_reranking"]),
                "p95_reranking_ms": nan_to_none(s["p95_reranking"]),
                "mean_total_ms": nan_to_none(s["mean_total"]),
                "std_total_ms": nan_to_none(s["std_total"]),
                "p95_total_ms": nan_to_none(s["p95_total"]),
                "reranking_overhead_pct": nan_to_none(
                    100.0 * (s["mean_total"] - s["mean_retrieval"]) / s["mean_total"]
                    if s["mean_total"] > 0
                    else 0.0
                ),
                "p_chr_auc": nan_to_none(s["p_chr_auc"])
                if s.get("p_chr_auc") is not None
                else None,
                "delta_ret": nan_to_none(s["delta_ret"])
                if s.get("delta_ret") is not None
                else None,
                "n_params": s.get("n_params"),
            }
            for s in all_stats_sorted
        ],
    }
    os.makedirs(args.plots_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved metrics to {args.output}")
    plot_latency_bars(
        all_stats_sorted, os.path.join(args.plots_dir, "latency_breakdown.png")
    )
    plot_latency_per_reranker(
        all_stats_sorted, os.path.join(args.plots_dir, "latency_per_reranker.png")
    )
    plot_latency_per_retriever(
        all_stats_sorted, os.path.join(args.plots_dir, "latency_per_retriever.png")
    )
