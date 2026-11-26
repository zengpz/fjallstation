"""A small example store: five texts, five chats and five tasks.

    fjallstation example ./store

The files are a part of the package, so the command needs no Docker and no network. The store has the format of
docs/data.md, and that document names the source of each example.
"""

import shutil
from pathlib import Path

STORE = Path(__file__).parent / "store"


def make(folder) -> Path:
    """Copy the example store into `folder`. Other sets in the folder stay as they are."""
    folder = Path(folder)
    for part in sorted(STORE.rglob("*.parquet")):
        target = folder / part.relative_to(STORE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(part, target)
    return folder
