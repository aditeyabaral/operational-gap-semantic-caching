"""Dataset statistics for the LangCache SentencePairs collection (Table: dataset-stats).

Loads the published redis/langcache-sentencepairs-{v1,v2,v3} datasets and prints the per-version
train / val / test split sizes as a rich table, reproducing the dataset-statistics table in the
paper. The per-source breakdown (Table: all-sources) is a provenance table produced by the dataset
curation scripts (src/sentencepairs/create_sentencepairs_v*.py), which merge the source datasets
listed there; it is not recomputed here because it requires the full curation build.

Usage:
    python -m src.analysis.dataset_stats
    python -m src.analysis.dataset_stats --versions v1 v2 v3
"""

import argparse
import sys

sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table

from src.reranker.util import load_langcache_sentencepairs_splits


def version_split_counts(version: str) -> dict:
    """Return {'train': n, 'val': n, 'test': n} for one published SentencePairs version."""
    train, val, test = load_langcache_sentencepairs_splits(
        subset_names={f"redis/langcache-sentencepairs-{version}": ["all"]},
        combine_train_and_val=False,
    )
    return {
        "train": len(train) if train is not None else 0,
        "val": len(val) if val is not None else 0,
        "test": len(test) if test is not None else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3"],
        help="Dataset versions to summarize (default: v1 v2 v3).",
    )
    args = parser.parse_args()

    console = Console()
    table = Table(
        title="LangCache SentencePairs dataset statistics by version",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Version", justify="left")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("Test", justify="right")

    for version in args.versions:
        print(f"Loading redis/langcache-sentencepairs-{version} ...", flush=True)
        counts = version_split_counts(version)
        table.add_row(
            version,
            f"{counts['train']:,}",
            f"{counts['val']:,}",
            f"{counts['test']:,}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
