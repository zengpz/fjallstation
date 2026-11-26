"""The data loaders: `Corpus()` and `Instruct()` read Parquet files of the store folder."""

import os
from pathlib import Path


def store() -> Path:
    """The folder that holds the data. The variable `FJALLSTATION_STORE` names it; without it, `./store` is used."""
    folder = Path(os.environ.get("FJALLSTATION_STORE") or "store")
    if not folder.is_dir():
        raise FileNotFoundError(f"No data folder at {folder.resolve()}. Set FJALLSTATION_STORE to the folder with the data (see docs/data.md).")
    return folder


def parts(name: str) -> list[str]:
    """The Parquet files of all sets of one object, for example `corpus/lessons/part-00000.parquet`."""
    files = sorted(str(p) for p in (store() / name).glob("*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"The data folder {store().resolve()} has no {name} data (see docs/data.md).")
    return files


def Instruct():
    """Chats for supervised fine-tuning.

    Returns a Hugging Face `datasets.Dataset`. Each row has `id` and `messages`.
    `messages` is a list of `{"role": ..., "content": ...}` items.
    """
    import datasets

    return datasets.load_dataset("parquet", data_files=parts("instruct"), split="train")


def Corpus():
    """Domain text for pretraining and mid-training.

    Returns a Hugging Face `datasets.IterableDataset`, the streaming form: a loop reads the rows from
    the files and does not load them all. Each row has `id`, `text` and `source`.
    """
    import datasets

    return datasets.load_dataset("parquet", data_files=parts("corpus"), split="train", streaming=True)
