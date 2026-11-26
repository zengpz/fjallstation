import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nekaise


def write(folder, rows):
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), folder / "part-00000.parquet")


def test_corpus_and_instruct_read_every_set_of_the_data_folder(tmp_path, monkeypatch):
    write(tmp_path / "corpus/a", [{"id": "1", "text": "A heat pump moves heat.", "source": "example"}])
    write(tmp_path / "corpus/b", [{"id": "2", "text": "A room loses heat.", "source": "example"}])
    write(tmp_path / "instruct/chats", [{"id": "c1", "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}])
    monkeypatch.setenv("NEKAISE_STORE", str(tmp_path))
    assert [doc["text"] for doc in nekaise.Corpus()] == ["A heat pump moves heat.", "A room loses heat."]
    chats = nekaise.Instruct()
    assert len(chats) == 1 and chats[0]["messages"][1] == {"role": "assistant", "content": "Hello"}


def test_a_missing_folder_or_a_missing_part_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NEKAISE_STORE", str(tmp_path / "nothing"))
    with pytest.raises(FileNotFoundError, match="NEKAISE_STORE"):
        nekaise.Corpus()
    monkeypatch.setenv("NEKAISE_STORE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no instruct data"):
        nekaise.Instruct()
