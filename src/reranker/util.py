import gc
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm


def load_langcache_sentencepairs_splits(
    subset_names: dict[str, str] | None = None,
    combine_train_and_val: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load train, val and test datasets from the LangCache Sentence Pairs dataset.

    Args:
        subset_names: Dictionary of dataset names and subset names to load. If not provided, all subsets will be loaded.
        combine_train_and_val: Whether to combine train and val datasets into a single train dataset.

    Returns:
        train_dataset: Train dataset, or None if no train split exists.
        val_dataset: Validation dataset, or None if no validation split exists or combine_train_and_val=True.
        test_dataset: Test dataset, or None if no test split exists.
    """
    if subset_names is None:
        subset_names = {"redis/langcache-sentencepairs-v3": ["all"]}
    train_datasets, val_datasets, test_datasets = [], [], []
    columns_to_keep = ["sentence1", "sentence2", "label"]

    for dataset_name, subsets in subset_names.items():
        for subset_name in subsets:
            dataset = load_dataset(dataset_name, subset_name)

            # Handle column name variations (sentence1/sentence2 vs sentence_a/sentence_b)
            # Check first available split to determine column names
            first_split = next(iter(dataset.values()))
            if "sentence_a" in first_split.column_names:
                dataset = dataset.rename_column("sentence_a", "sentence1")
                dataset = dataset.rename_column("sentence_b", "sentence2")

            # Select columns for each split
            try:
                train_datasets.append(dataset["train"].select_columns(columns_to_keep))
            except KeyError:
                pass
            try:
                val_datasets.append(
                    dataset["validation"].select_columns(columns_to_keep)
                )
            except KeyError:
                pass
            try:
                test_datasets.append(dataset["test"].select_columns(columns_to_keep))
            except KeyError:
                pass

    # Concatenate datasets
    train_dataset = concatenate_datasets(train_datasets) if train_datasets else None
    val_dataset = concatenate_datasets(val_datasets) if val_datasets else None
    test_dataset = concatenate_datasets(test_datasets) if test_datasets else None

    # Combine train and val datasets if specified
    if combine_train_and_val and val_dataset is not None and train_dataset is not None:
        train_dataset = concatenate_datasets([train_dataset, val_dataset])
        val_dataset = None  # Set to None since we combined it into train

    return train_dataset, val_dataset, test_dataset


def to_infonce(
    ds: Dataset,
    *,
    num_negatives: int = 3,
    seed: int = 42,
    cache_dir: str | None = None,
) -> Dataset:
    """Convert a sentence-pairs dataset (sentence1, sentence2, label in {0,1})
    into InfoNCE-ready examples with columns: anchor, positive, negative_1, negative_2, ..., negative_n.

    Args:
        ds: HF Dataset with columns ['sentence1', 'sentence2', 'label'].
        num_negatives: Number of negatives to sample per (anchor, positive) pair.
        seed: RNG seed for deterministic negative sampling.
        cache_dir: Directory to store temporary chunk files. If None, uses system temp dir.

    Returns:
        Dataset with columns: 'anchor', 'positive', 'negative_1', ..., 'negative_{num_negatives}'.
    """
    np_rng = np.random.RandomState(seed)

    with tqdm(total=3, desc="Loading data columns") as pbar:
        s1_list = ds["sentence1" if "sentence1" in ds.column_names else "sentence_a"]
        pbar.update(1)
        s2_list = ds["sentence2" if "sentence2" in ds.column_names else "sentence_b"]
        pbar.update(1)
        y_list = np.array(ds["label"], dtype=np.int32)
        pbar.update(1)

    valid_mask = np.array(
        [
            s1 is not None and s2 is not None
            for s1, s2 in tqdm(
                zip(s1_list, s2_list),
                total=len(s1_list),
                desc="Filtering None values",
            )
        ]
    )
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"Warning: Filtering out {n_invalid} pairs with None values")
        s1_list = [
            s1
            for s1, valid in tqdm(
                zip(s1_list, valid_mask), total=len(valid_mask), desc="Filtering s1"
            )
            if valid
        ]
        s2_list = [
            s2
            for s2, valid in tqdm(
                zip(s2_list, valid_mask), total=len(valid_mask), desc="Filtering s2"
            )
            if valid
        ]
        y_list = y_list[valid_mask]

    n_pairs = len(s1_list)
    print(f"Building sentence vocabulary from {n_pairs} pairs...")

    with tqdm(total=2, desc="Converting to numpy arrays") as pbar:
        s1_arr = np.array(s1_list, dtype=object)
        pbar.update(1)
        s2_arr = np.array(s2_list, dtype=object)
        pbar.update(1)
    del s1_list, s2_list, valid_mask
    gc.collect()

    print("Finding unique sentences (vectorized)...")
    all_sentences_concat = np.concatenate([s1_arr, s2_arr])
    all_sentences, inverse_indices = np.unique(
        all_sentences_concat, return_inverse=True
    )

    s1_ids = inverse_indices[:n_pairs].astype(np.int32)
    s2_ids = inverse_indices[n_pairs:].astype(np.int32)

    n_sent = len(all_sentences)
    print(f"Found {n_sent} unique sentences")

    del s1_arr, s2_arr, all_sentences_concat, inverse_indices
    gc.collect()

    print("Building relationship graphs...")
    positives = defaultdict(list)
    negatives = defaultdict(list)

    pos_mask = y_list == 1
    neg_mask = y_list == 0

    print("Building positive edges...")
    pos_indices = np.where(pos_mask)[0]
    for i in tqdm(pos_indices, desc="Positive edges"):
        ia, ib = s1_ids[i], s2_ids[i]
        positives[ia].append(ib)
        positives[ib].append(ia)

    print("Building negative edges...")
    neg_indices = np.where(neg_mask)[0]
    for i in tqdm(neg_indices, desc="Negative edges"):
        ia, ib = s1_ids[i], s2_ids[i]
        negatives[ia].append(ib)
        negatives[ib].append(ia)

    del s1_ids, s2_ids, y_list
    gc.collect()

    print("Converting to numpy arrays...")
    pos_arrays = {k: np.array(v, dtype=np.int32) for k, v in positives.items()}
    neg_arrays = {k: np.array(v, dtype=np.int32) for k, v in negatives.items()}

    del positives, negatives, pos_mask, neg_mask, pos_indices, neg_indices
    gc.collect()

    print("Counting output examples...")
    all_anchors = list(set(pos_arrays.keys()) | set(neg_arrays.keys()))
    n_out = sum(max(1, len(pos_arrays.get(a, []))) for a in all_anchors)
    print(f"Total output examples: {n_out}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="infonce_", dir=cache_dir)
    else:
        temp_dir = tempfile.mkdtemp(prefix="infonce_")
    print(f"Using temporary directory: {temp_dir}")

    try:
        print(f"Creating memory-mapped arrays for {n_out} examples...")
        anchors_out = np.memmap(
            os.path.join(temp_dir, "anchors.mmap"),
            dtype=np.int32,
            mode="w+",
            shape=(n_out,),
        )
        positives_out = np.memmap(
            os.path.join(temp_dir, "positives.mmap"),
            dtype=np.int32,
            mode="w+",
            shape=(n_out,),
        )
        negatives_out = np.memmap(
            os.path.join(temp_dir, "negatives.mmap"),
            dtype=np.int32,
            mode="w+",
            shape=(n_out, num_negatives),
        )

        all_ids = np.arange(n_sent, dtype=np.int32)

        def sample_negatives_fast(blocked_set: set, size: int) -> np.ndarray:
            """Fast rejection sampling without array copies. Returns unique samples."""
            if len(blocked_set) >= n_sent:
                fallback = next((i for i in range(n_sent) if i not in blocked_set), 0)
                return np.full(size, fallback, dtype=np.int32)

            n_available = n_sent - len(blocked_set)

            if len(blocked_set) < n_sent * 0.1:
                if n_available < size:
                    result = np.empty(size, dtype=np.int32)
                    for i in range(size):
                        for _ in range(100):
                            candidate = np_rng.randint(0, n_sent)
                            if candidate not in blocked_set:
                                result[i] = candidate
                                break
                        else:
                            result[i] = next(
                                (j for j in range(n_sent) if j not in blocked_set), 0
                            )
                    return result
                result = np.empty(size, dtype=np.int32)
                sampled = set()
                for i in range(size):
                    for _ in range(1000):
                        candidate = np_rng.randint(0, n_sent)
                        if candidate not in blocked_set and candidate not in sampled:
                            result[i] = candidate
                            sampled.add(candidate)
                            break
                    else:
                        result[i] = next(
                            (
                                j
                                for j in range(n_sent)
                                if j not in blocked_set and j not in sampled
                            ),
                            0,
                        )
                        sampled.add(result[i])
                return result
            mask = np.ones(n_sent, dtype=bool)
            mask[list(blocked_set)] = False
            candidates = all_ids[mask]  # noqa: F821
            if len(candidates) >= size:
                return np_rng.choice(candidates, size=size, replace=False)
            if len(candidates) > 0:
                return np_rng.choice(candidates, size=size, replace=True)
            fallback = next((i for i in range(n_sent) if i not in blocked_set), 0)
            return np.full(size, fallback, dtype=np.int32)

        idx = 0
        print("Generating triplets...")
        for a in tqdm(all_anchors, desc="Processing anchors"):
            pos_ids = pos_arrays.get(a, np.array([], dtype=np.int32))
            neg_ids = neg_arrays.get(a, np.array([], dtype=np.int32))

            if len(pos_ids) == 0:
                anchors_out[idx] = a
                positives_out[idx] = a

                if len(neg_ids) >= num_negatives:
                    negatives_out[idx] = np_rng.choice(
                        neg_ids, size=num_negatives, replace=False
                    )
                else:
                    neg_sample = sample_negatives_fast({a}, num_negatives)
                    negatives_out[idx] = neg_sample
                idx += 1
            else:
                n_pos = len(pos_ids)

                if len(neg_ids) >= num_negatives:
                    neg_choices = np.array(
                        [
                            np_rng.choice(neg_ids, size=num_negatives, replace=False)
                            for _ in range(n_pos)
                        ]
                    )
                else:
                    blocked = {a} | set(pos_ids.tolist())
                    neg_choices = np.array(
                        [
                            sample_negatives_fast(blocked, num_negatives)
                            for _ in range(n_pos)
                        ]
                    )

                anchors_out[idx : idx + n_pos] = a
                positives_out[idx : idx + n_pos] = pos_ids
                negatives_out[idx : idx + n_pos] = neg_choices
                idx += n_pos

        print("Flushing data to disk...")
        anchors_out.flush()
        positives_out.flush()
        negatives_out.flush()

        del pos_arrays, neg_arrays, all_ids
        gc.collect()

        # ============================================================================
        # KEY OPTIMIZATION: Write directly to a single Parquet file instead of
        # creating separate Arrow files and concatenating them
        # ============================================================================
        print("Writing directly to Parquet file (no concatenation needed)...")

        parquet_path = os.path.join(temp_dir, "dataset.parquet")
        chunk_size = 100_000  # Even smaller chunks for writing
        n_chunks = (idx + chunk_size - 1) // chunk_size

        # Define schema
        schema_fields = [
            pa.field("anchor", pa.string()),
            pa.field("positive", pa.string()),
        ]
        for i in range(num_negatives):
            schema_fields.append(pa.field(f"negative_{i + 1}", pa.string()))
        schema = pa.schema(schema_fields)

        # Write to Parquet file with streaming
        with pq.ParquetWriter(parquet_path, schema) as writer:
            for i in tqdm(range(n_chunks), desc="Writing to Parquet"):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, idx)

                # Convert IDs to strings
                anchor_texts = all_sentences[anchors_out[start_idx:end_idx]].tolist()
                positive_texts = all_sentences[
                    positives_out[start_idx:end_idx]
                ].tolist()
                negative_texts_2d = all_sentences[negatives_out[start_idx:end_idx]]

                # Build arrays for PyArrow
                arrays = [
                    pa.array(anchor_texts, type=pa.string()),
                    pa.array(positive_texts, type=pa.string()),
                ]
                for neg_idx in range(num_negatives):
                    arrays.append(
                        pa.array(
                            negative_texts_2d[:, neg_idx].tolist(), type=pa.string()
                        )
                    )

                # Create table and write
                batch = pa.Table.from_arrays(arrays, schema=schema)
                writer.write_table(batch)

                # Free memory immediately
                del anchor_texts, positive_texts, negative_texts_2d, arrays, batch
                gc.collect()

        del all_sentences, anchors_out, positives_out, negatives_out
        gc.collect()

        # Load the final dataset from Parquet
        print("Loading final dataset from Parquet...")
        result = Dataset.from_parquet(parquet_path)

        return result

    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")
