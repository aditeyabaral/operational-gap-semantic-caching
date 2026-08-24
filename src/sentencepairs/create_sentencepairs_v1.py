import multiprocessing
import os

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)


def convert_label_to_int(example):
    """Map the label to an integer.

    For the PIT-2015 dataset, we follow this convention:

    train:
        paraphrases: (3, 2) (4, 1) (5, 0)
        non-paraphrases: (1, 4) (0, 5)
        debatable: (2, 3)  which you may discard if training binary classifier
    """
    try:
        example["label"] = int(example["label"])
    except Exception:
        if example["label"] in ["(3, 2)", "(4, 1)", "(5, 0)"]:
            example["label"] = 1
        elif example["label"] in ["(1, 4)", "(0, 5)", "(2, 3)"]:
            example["label"] = 0
    return example


def standardize_labels(dataset):
    dataset = dataset.map(convert_label_to_int, num_proc=multiprocessing.cpu_count())
    dataset = dataset.cast(
        Features(
            {
                "sentence1": Value("string"),
                "sentence2": Value("string"),
                "label": Value("int8"),
            }
        ),
        num_proc=multiprocessing.cpu_count(),
    )
    return dataset


def remove_null_examples(dataset):
    return dataset.filter(
        lambda x: x.get("sentence1") is not None and x.get("sentence2") is not None,
        num_proc=multiprocessing.cpu_count(),
    )


def load_paws_dataset():
    dataset = load_dataset("google-research-datasets/paws", "unlabeled_final")
    train_dataset = dataset["train"].select_columns(["sentence1", "sentence2", "label"])
    test_dataset = dataset["validation"].select_columns(
        ["sentence1", "sentence2", "label"]
    )
    train_dataset = standardize_labels(train_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, None, test_dataset


def load_mrpc_dataset():
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    train_dataset = dataset["train"].select_columns(["sentence1", "sentence2", "label"])
    val_dataset = dataset["validation"].select_columns(
        ["sentence1", "sentence2", "label"]
    )
    test_dataset = dataset["test"].select_columns(["sentence1", "sentence2", "label"])
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_qqp_dataset():
    dataset = load_dataset("nyu-mll/glue", "qqp")
    train_dataset = dataset["train"].rename_columns(
        {"question1": "sentence1", "question2": "sentence2"}
    )
    test_dataset = dataset["validation"].rename_columns(
        {"question1": "sentence1", "question2": "sentence2"}
    )
    train_dataset = train_dataset.select_columns(["sentence1", "sentence2", "label"])
    test_dataset = test_dataset.select_columns(["sentence1", "sentence2", "label"])
    train_dataset = standardize_labels(train_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, None, test_dataset


def load_stsb_dataset():
    dataset = load_dataset("glue", "stsb")
    train_dataset = dataset["train"].select_columns(["sentence1", "sentence2", "label"])
    val_dataset = dataset["validation"].select_columns(
        ["sentence1", "sentence2", "label"]
    )
    test_dataset = dataset["test"].select_columns(["sentence1", "sentence2", "label"])

    for split_dataset, df_name in [
        (train_dataset, "train"),
        (val_dataset, "val"),
        (test_dataset, "test"),
    ]:
        df = split_dataset.to_pandas()
        df["label"] = (df["label"] > 3.5).astype(int)
        if df_name == "train":
            train_dataset = Dataset.from_pandas(df)
        elif df_name == "val":
            val_dataset = Dataset.from_pandas(df)
        else:
            test_dataset = Dataset.from_pandas(df)

    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_parade_dataset(dir: str = "data/parade"):
    column_map = {
        "Definition1": "sentence1",
        "Definition2": "sentence2",
        "Binary labels": "label",
    }
    train_df = pd.read_csv(os.path.join(dir, "PARADE_train.txt"), sep="\t").rename(
        columns=column_map
    )[["sentence1", "sentence2", "label"]]
    val_df = pd.read_csv(os.path.join(dir, "PARADE_validation.txt"), sep="\t").rename(
        columns=column_map
    )[["sentence1", "sentence2", "label"]]
    test_df = pd.read_csv(os.path.join(dir, "PARADE_test.txt"), sep="\t").rename(
        columns=column_map
    )[["sentence1", "sentence2", "label"]]
    train_dataset = standardize_labels(Dataset.from_pandas(train_df))
    val_dataset = standardize_labels(Dataset.from_pandas(val_df))
    test_dataset = standardize_labels(Dataset.from_pandas(test_df))
    return train_dataset, val_dataset, test_dataset


def load_pit2015_dataset(dir: str = "data/pit2015"):
    col_names = [
        "Topic_Id",
        "Topic_Name",
        "Sent_1",
        "Sent_2",
        "Label",
        "Sent_1_tag",
        "Sent_2_tag",
    ]
    column_map = {"Sent_1": "sentence1", "Sent_2": "sentence2", "Label": "label"}
    train_df = pd.read_csv(
        os.path.join(dir, "train.data"), sep="\t", header=None, names=col_names
    ).rename(columns=column_map)[["sentence1", "sentence2", "label"]]
    val_df = pd.read_csv(
        os.path.join(dir, "dev.data"), sep="\t", header=None, names=col_names
    ).rename(columns=column_map)[["sentence1", "sentence2", "label"]]
    test_df = pd.read_csv(
        os.path.join(dir, "test.data"), sep="\t", header=None, names=col_names
    ).rename(columns=column_map)[["sentence1", "sentence2", "label"]]
    test_df["label"] = test_df["label"].apply(lambda x: 1 if x in [4, 5] else 0)
    train_dataset = standardize_labels(Dataset.from_pandas(train_df))
    val_dataset = standardize_labels(Dataset.from_pandas(val_df))
    test_dataset = standardize_labels(Dataset.from_pandas(test_df))
    return train_dataset, val_dataset, test_dataset


def load_apt_dataset(dir: str = "data/apt"):
    column_map = {"text_a": "sentence1", "text_b": "sentence2", "labels": "label"}
    train_df = pd.read_csv(os.path.join(dir, "train.tsv"), sep="\t").rename(
        columns=column_map
    )[["sentence1", "sentence2", "label"]]
    test_df = pd.read_csv(os.path.join(dir, "test.tsv"), sep="\t").rename(
        columns=column_map
    )[["sentence1", "sentence2", "label"]]
    train_dataset = standardize_labels(Dataset.from_pandas(train_df))
    test_dataset = standardize_labels(Dataset.from_pandas(test_df))
    return train_dataset, None, test_dataset


def load_sick_dataset(dir: str = "data/sick"):
    df = pd.read_csv(os.path.join(dir, "SICK.txt"), sep="\t").rename(
        columns={
            "sentence_A": "sentence1",
            "sentence_B": "sentence2",
            "entailment_label": "label",
        }
    )
    splits = {"train": "TRAIN", "val": "TRIAL", "test": "TEST"}
    datasets = {}
    for name, split in splits.items():
        split_df = df[df["SemEval_set"] == split][
            ["sentence1", "sentence2", "label"]
        ].copy()
        split_df["label"] = (split_df["label"] != "CONTRADICTION").astype(int)
        ds = Dataset.from_pandas(split_df)
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])
        datasets[name] = standardize_labels(ds)
    return datasets["train"], datasets["val"], datasets["test"]


if __name__ == "__main__":
    callables = {
        "paws": load_paws_dataset,
        "mrpc": load_mrpc_dataset,
        "qqp": load_qqp_dataset,
        "parade": load_parade_dataset,
        "pit2015": load_pit2015_dataset,
        "apt": load_apt_dataset,
        "stsb": load_stsb_dataset,
        "sick": load_sick_dataset,
    }

    dataset = DatasetDict()
    train_datasets, val_datasets, test_datasets = [], [], []

    for name, callable in callables.items():
        train_dataset, val_dataset, test_dataset = callable()
        train_dataset = remove_null_examples(train_dataset)

        dataset[name] = DatasetDict({"train": train_dataset})
        if val_dataset is not None:
            val_dataset = remove_null_examples(val_dataset)
            dataset[name]["validation"] = val_dataset
        if test_dataset is not None:
            test_dataset = remove_null_examples(test_dataset)
            dataset[name]["test"] = test_dataset

        dataset[name].push_to_hub(
            "redis/langcache-sentencepairs-v1",
            config_name=name,
            max_shard_size="500MB",
        )

        train_dataset = train_dataset.add_column("source", [name] * len(train_dataset))
        train_dataset = train_dataset.add_column(
            "source_idx", list(range(len(train_dataset)))
        )
        train_datasets.append(train_dataset)

        if val_dataset is not None:
            val_dataset = val_dataset.add_column("source", [name] * len(val_dataset))
            val_dataset = val_dataset.add_column(
                "source_idx", list(range(len(val_dataset)))
            )
            val_datasets.append(val_dataset)

        if test_dataset is not None:
            test_dataset = test_dataset.add_column("source", [name] * len(test_dataset))
            test_dataset = test_dataset.add_column(
                "source_idx", list(range(len(test_dataset)))
            )
            test_datasets.append(test_dataset)

    print(dataset)

    merged_dataset = DatasetDict(
        {
            "train": concatenate_datasets(train_datasets),
            "validation": concatenate_datasets(val_datasets),
            "test": concatenate_datasets(test_datasets),
        }
    )

    def remove_duplicates(dataset):
        df = dataset.to_pandas().drop_duplicates(subset=["sentence1", "sentence2"])
        unique_dataset = Dataset.from_pandas(df)
        if "__index_level_0__" in unique_dataset.column_names:
            unique_dataset = unique_dataset.remove_columns(["__index_level_0__"])
        return unique_dataset

    print(
        f"Before deduplication - Train: {len(merged_dataset['train'])}, Val: {len(merged_dataset['validation'])}, Test: {len(merged_dataset['test'])}"
    )
    merged_dataset["train"] = remove_duplicates(merged_dataset["train"])
    merged_dataset["validation"] = remove_duplicates(merged_dataset["validation"])
    merged_dataset["test"] = remove_duplicates(merged_dataset["test"])
    print(
        f"After deduplication - Train: {len(merged_dataset['train'])}, Val: {len(merged_dataset['validation'])}, Test: {len(merged_dataset['test'])}"
    )

    for split in ["train", "validation", "test"]:
        total = len(merged_dataset[split])
        padding = len(str(total - 1))
        ids = [f"langcache_{split}_{idx:0{padding}d}" for idx in range(total)]
        merged_dataset[split] = merged_dataset[split].add_column("id", ids)

    for split in ["train", "validation", "test"]:
        merged_dataset[split] = merged_dataset[split].cast(
            Features(
                {
                    "id": Value("string"),
                    "source_idx": Value("int32"),
                    "source": Value("string"),
                    "sentence1": Value("string"),
                    "sentence2": Value("string"),
                    "label": Value("int8"),
                }
            ),
            num_proc=multiprocessing.cpu_count() // 2,
        )
        merged_dataset[split] = merged_dataset[split].select_columns(
            ["id", "source_idx", "source", "sentence1", "sentence2", "label"]
        )

    print(merged_dataset)
    merged_dataset.push_to_hub(
        "redis/langcache-sentencepairs-v1",
        config_name="all",
        max_shard_size="500MB",
    )
