import multiprocessing
import os
import re
from itertools import combinations

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
    """Remove examples with null values in the sentence1 or sentence2 columns."""
    return dataset.filter(
        lambda x: x.get("sentence1") is not None and x.get("sentence2") is not None,
        num_proc=multiprocessing.cpu_count(),
    )


def load_paws_dataset():
    """Load the PAWS dataset from the Hugging Face Hub.

    Subset: unlabeled_final
    Splits: train, test
    """
    dataset = load_dataset("google-research-datasets/paws", "unlabeled_final")
    train_dataset = dataset["train"].select_columns(["sentence1", "sentence2", "label"])
    test_dataset = dataset["validation"].select_columns(
        ["sentence1", "sentence2", "label"]
    )
    train_dataset = standardize_labels(train_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, None, test_dataset


def load_mrpc_dataset():
    """Load the MRPC dataset from the Hugging Face Hub.

    Splits: train, validation, test
    """
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
    """Load the QQP dataset from the Hugging Face Hub.

    Splits: train, test
    """
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
    """Load the STS-B dataset from the Hugging Face Hub.

    Labels are binarized: scores > 3.5 are positive, others negative.

    Splits: train, validation, test
    """
    dataset = load_dataset("glue", "stsb")
    train_dataset = dataset["train"].select_columns(["sentence1", "sentence2", "label"])
    val_dataset = dataset["validation"].select_columns(
        ["sentence1", "sentence2", "label"]
    )
    test_dataset = dataset["test"].select_columns(["sentence1", "sentence2", "label"])

    train_df = train_dataset.to_pandas()
    train_df["label"] = (train_df["label"] > 3.5).astype(int)
    train_dataset = Dataset.from_pandas(train_df)

    val_df = val_dataset.to_pandas()
    val_df["label"] = (val_df["label"] > 3.5).astype(int)
    val_dataset = Dataset.from_pandas(val_df)

    test_df = test_dataset.to_pandas()
    test_df["label"] = (test_df["label"] > 3.5).astype(int)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_opusparcus_dataset(dir: str = "data/opusparcus"):
    """Load the OpusParCus dataset from the local directory.

    Only English examples are retained. Train split filters by quality >= 90.

    Splits: train, validation, test
    """
    train_path = os.path.join(dir, "train_en.70.jsonl")
    val_path = os.path.join(dir, "validation.jsonl")
    test_path = os.path.join(dir, "test.jsonl")

    def should_keep(example, filter_quality=True):
        try:
            return not (
                (filter_quality and example.get("quality", 0) < 90)
                or example["lang"] != "en"
            )
        except Exception:
            return False

    def process_example(example):
        try:
            if "annot_score" in example:
                label = 1.0 if example["annot_score"] >= 3.0 else 0.0
            else:
                label = 1.0
            return {
                "sentence1": example["sent1"],
                "sentence2": example["sent2"],
                "label": label,
            }
        except Exception:
            return {"sentence1": "", "sentence2": "", "label": 0}

    train_raw = load_dataset("json", data_files=train_path, split="train")
    val_raw = load_dataset("json", data_files=val_path, split="train")
    test_raw = load_dataset("json", data_files=test_path, split="train")

    train_dataset = train_raw.filter(
        lambda x: should_keep(x, filter_quality=True),
        num_proc=multiprocessing.cpu_count(),
    ).map(process_example, num_proc=multiprocessing.cpu_count())

    val_dataset = val_raw.filter(
        lambda x: should_keep(x, filter_quality=False),
        num_proc=multiprocessing.cpu_count(),
    ).map(process_example, num_proc=multiprocessing.cpu_count())

    test_dataset = test_raw.filter(
        lambda x: should_keep(x, filter_quality=False),
        num_proc=multiprocessing.cpu_count(),
    ).map(process_example, num_proc=multiprocessing.cpu_count())

    train_dataset = standardize_labels(
        train_dataset.select_columns(["sentence1", "sentence2", "label"])
    )
    val_dataset = standardize_labels(
        val_dataset.select_columns(["sentence1", "sentence2", "label"])
    )
    test_dataset = standardize_labels(
        test_dataset.select_columns(["sentence1", "sentence2", "label"])
    )
    return train_dataset, val_dataset, test_dataset


def load_parade_dataset(dir: str = "data/parade"):
    """Load the PARADE dataset from the local directory.

    Splits: train, validation, test
    """
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


def load_ttic31190_dataset(dir: str = "data/ttic31190"):
    """Load the TTIC-31190 dataset from the local directory.

    Train split contains only positive pairs (no labels in source file).

    Splits: train, validation, test
    """
    train_df = pd.read_csv(
        os.path.join(dir, "train.tsv"),
        sep="\t",
        header=None,
        names=["sentence1", "sentence2"],
        quoting=3,
        encoding="utf-8",
    )
    train_df["label"] = 1  # All training examples are positive pairs
    val_df = pd.read_csv(
        os.path.join(dir, "dev.tsv"),
        sep="\t",
        header=None,
        names=["sentence1", "sentence2", "label"],
        quoting=3,
        encoding="utf-8",
    )
    test_df = pd.read_csv(
        os.path.join(dir, "devtest.tsv"),
        sep="\t",
        header=None,
        names=["sentence1", "sentence2", "label"],
        quoting=3,
        encoding="utf-8",
    )
    train_dataset = standardize_labels(Dataset.from_pandas(train_df))
    val_dataset = standardize_labels(Dataset.from_pandas(val_df))
    test_dataset = standardize_labels(Dataset.from_pandas(test_df))
    return train_dataset, val_dataset, test_dataset


def load_pit2015_dataset(dir: str = "data/pit2015"):
    """Load the PIT-2015 dataset from the local directory.

    Splits: train, validation, test
    """
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
    """Load the APT dataset from the local directory.

    Splits: train, test
    """
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
    """Load the SICK dataset from the local directory.

    Labels are binarized: CONTRADICTION -> 0, all others -> 1.

    Splits: train, validation, test
    """
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


def load_tapaco_dataset():
    """Load the TAPACO dataset from the Hugging Face Hub.

    English subset only. Pairs are generated from paraphrase clusters.

    Splits: train
    """
    dataset = load_dataset("community-datasets/tapaco", "en", split="train")
    grouped = dataset.to_pandas().groupby("paraphrase_set_id")["paraphrase"].apply(list)
    cluster_dataset = Dataset.from_dict({"paraphrases": grouped.tolist()})

    def make_pairs(batch):
        results = {"sentence1": [], "sentence2": [], "label": []}
        for sentences in batch["paraphrases"]:
            if len(sentences) < 2:
                continue
            for s1, s2 in combinations(sentences, 2):
                results["sentence1"].append(s1)
                results["sentence2"].append(s2)
                results["label"].append(1.0)
        return results

    pairs = cluster_dataset.map(
        make_pairs,
        batched=True,
        remove_columns=["paraphrases"],
        num_proc=multiprocessing.cpu_count(),
    )
    pairs = standardize_labels(pairs)
    return pairs, None, None


def load_paraphrase_collections_dataset():
    """Load the Paraphrase Collections dataset from the Hugging Face Hub.

    All examples are positive pairs.

    Splits: train
    """
    dataset = load_dataset("xwjzds/paraphrase_collections", split="train")
    dataset = dataset.rename_columns({"input": "sentence1", "output": "sentence2"})
    dataset = dataset.add_column("label", [1] * len(dataset))
    dataset = standardize_labels(
        dataset.select_columns(["sentence1", "sentence2", "label"])
    )
    return dataset, None, None


def load_chatgpt_paraphrases_dataset():
    """Load the ChatGPT Paraphrases dataset from the Hugging Face Hub.

    All examples are positive pairs.

    Splits: train
    """
    dataset = load_dataset("sharad/chatgpt-paraphrases-simple", split="train")
    dataset = dataset.rename_columns({"s1": "sentence1", "s2": "sentence2"})
    dataset = dataset.add_column("label", [1] * len(dataset))
    dataset = standardize_labels(
        dataset.select_columns(["sentence1", "sentence2", "label"])
    )
    return dataset, None, None


def load_paranmt_dataset(dir: str = "data/paranmt/para-nmt-5m-processed"):
    """Load the ParaNMT-5M dataset from the local directory.

    All examples are positive pairs.

    Splits: train
    """
    df = pd.read_csv(
        os.path.join(dir, "para-nmt-5m-processed.txt"),
        sep="\t",
        header=None,
        names=["sentence1", "sentence2"],
    )
    df["label"] = 1.0
    train_dataset = standardize_labels(Dataset.from_pandas(df))
    return train_dataset, None, None


def load_task275_enhanced_wsc_paraphrase_generation_dataset():
    """Load the Task 275 Enhanced WSC Paraphrase Generation dataset from the Hugging Face Hub.

    Splits: train, validation, test
    """

    def process_example(example):
        input_text = example["input"]
        matches = re.findall(
            r"Input:\s*sentence:\s*(.*?)\s*aspect:", input_text, re.DOTALL
        )
        sentence1 = matches[-1].strip() if matches else ""
        sentence2 = (
            example["output"][0].strip()
            if example["output"] and len(example["output"]) > 0
            else ""
        )
        return {"sentence1": sentence1, "sentence2": sentence2, "label": 1.0}

    dataset = load_dataset("Lots-of-LoRAs/task275_enhanced_wsc_paraphrase_generation")

    def process_split(split):
        ds = dataset[split].map(process_example, num_proc=multiprocessing.cpu_count())
        ds = ds.select_columns(["sentence1", "sentence2", "label"])
        ds = ds.filter(
            lambda x: x["sentence1"].strip() != "" and x["sentence2"].strip() != "",
            num_proc=multiprocessing.cpu_count(),
        )
        return standardize_labels(ds)

    return process_split("train"), process_split("valid"), process_split("test")


def load_llm_paraphrases_dataset():
    """Load the LLM Paraphrases dataset from the Hugging Face Hub.

    Splits: train, test
    """
    dataset = load_dataset("redis/llm-paraphrases")
    train_dataset = dataset["train"].rename_columns(
        {"sentence_a": "sentence1", "sentence_b": "sentence2"}
    )
    test_dataset = dataset["test"].rename_columns(
        {"sentence_a": "sentence1", "sentence_b": "sentence2"}
    )
    train_dataset = standardize_labels(
        train_dataset.select_columns(["sentence1", "sentence2", "label"])
    )
    test_dataset = standardize_labels(
        test_dataset.select_columns(["sentence1", "sentence2", "label"])
    )
    return train_dataset, None, test_dataset


def load_parabank2_dataset(dir: str = "data/parabank2"):
    """Load the ParaBank2 dataset from the local directory.

    All examples are positive pairs.

    Splits: train
    """
    dataset = load_dataset(
        "csv",
        data_files=os.path.join(dir, "parabank2.tsv"),
        sep="\t",
        column_names=["score", "sentence1", "sentence2"],
        usecols=[1, 2],
    )["train"]
    dataset = dataset.add_column("label", [1] * len(dataset))
    dataset = standardize_labels(dataset)
    return dataset, None, None


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
        "ttic31190": load_ttic31190_dataset,
        "tapaco": load_tapaco_dataset,
        "paraphrase-collections": load_paraphrase_collections_dataset,
        "chatgpt-paraphrases": load_chatgpt_paraphrases_dataset,
        "opusparcus": load_opusparcus_dataset,
        "paranmt5m": lambda: load_paranmt_dataset("data/paranmt/para-nmt-5m-processed"),
        "task275-enhanced-wsc-paraphrase-generation": load_task275_enhanced_wsc_paraphrase_generation_dataset,
        "llm-paraphrases": load_llm_paraphrases_dataset,
        "parabank2": load_parabank2_dataset,
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
            "redis/langcache-sentencepairs-v3",
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
        """Remove duplicate (sentence1, sentence2) pairs."""
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
        "redis/langcache-sentencepairs-v3",
        config_name="all",
        max_shard_size="500MB",
    )
