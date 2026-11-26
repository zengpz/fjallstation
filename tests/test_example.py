import json
import math

import pyarrow.parquet as pq
import pytest

import fjallstation
from fjallstation import example, image, score
from fjallstation import server as app
from fjallstation.__main__ import main

TASKS = ["example-heated-room", "example-heating-curve", "example-pid-controller", "example-return-pipe", "example-heat-pump"]


def test_the_example_store_has_five_texts_five_chats_and_five_tasks(tmp_path, monkeypatch):
    store = example.make(tmp_path / "store")
    monkeypatch.setenv("FJALLSTATION_STORE", str(store))
    texts, chats = list(fjallstation.Corpus()), fjallstation.Instruct()
    assert [t["id"] for t in texts] == [f"example-text-{i}" for i in range(1, 6)] and all(len(t["text"]) > 500 and t["source"] == "generated" for t in texts)
    assert len(chats) == 5 and all([m["role"] for m in chat["messages"]] == ["user", "assistant"] for chat in chats)
    rows = pq.read_table(store / "env/tasks-example/part-00000.parquet").to_pylist()
    assert [r["id"] for r in rows] == TASKS and "reference_code" not in rows[0]                 # no solution is in the example store
    by_id = {r["id"]: r for r in rows}
    assert by_id["example-heated-room"]["files"] is None and by_id["example-heated-room"]["credit"] is None
    for name, doi in (("example-return-pipe", "zenodo.23127439"), ("example-heat-pump", "zenodo.21157950")):      # tasks from open datasets carry their credit
        row = by_id[name]
        assert sorted(json.loads(row["files"])) == ["calibration.txt", "evaluation.txt"] and doi in row["credit"] and "CC BY 4.0" in row["credit"]
        assert row["prompt"].rstrip().endswith(row["credit"])
    assert json.loads(by_id["example-heating-curve"]["spec"])["libraries"] == ["Buildings 13.0.0"]


def test_the_server_reads_the_example_tasks_and_scores_a_result(tmp_path):
    tasks = app.load_tasks(example.make(tmp_path / "store"))
    assert sorted(tasks) == sorted(TASKS)
    station = app.Station(tasks, tmp_path / "rewards", seed=0)
    task_file = station.sample("example-heat-pump", "http://here/v1/results")
    assert task_file["image"] == image.IMAGE and sorted(task_file["files"]) == ["calibration.txt", "evaluation.txt"]
    reference = tasks["example-heat-pump"]["reference_outputs"]["E_el"]
    assert math.isnan(reference[0]) and not math.isnan(reference[-1])                           # only the last days are compared
    right = {"ticket": task_file["ticket"], "image": image.IMAGE, "status": "simulated", "values": {"E_el": [0.0 if math.isnan(v) else v for v in reference]}}
    assert station.result(right)["reward"] == 1.0
    task_file = station.sample("example-heated-room", "http://here/v1/results")
    wrong = {"ticket": task_file["ticket"], "image": image.IMAGE, "status": "simulated", "values": {"T_room": [10.0] * 145}}
    assert station.result(wrong) == {"reward": 0.0, "omc_reward": 0.3, "status": "outputs differ", "task": "example-heated-room", "image": image.IMAGE}


def test_the_example_goes_into_a_folder_that_has_other_sets(tmp_path):
    other = tmp_path / "store/corpus/mine"
    other.mkdir(parents=True)
    (other / "part-00000.parquet").write_bytes(b"x")
    example.make(tmp_path / "store")
    assert (other / "part-00000.parquet").read_bytes() == b"x" and (tmp_path / "store/corpus/example/part-00000.parquet").exists()


def test_a_task_file_for_one_try_has_no_ticket_and_no_reference():
    spec = {"model_name": "Room", "msl": "4.1.0", "libraries": [], "eval": {"stop_time": 10, "interval": 5, "tolerance": 1e-6, "outputs": ["T"]}}
    assert image.task_file(spec, files={"a.txt": "1"}) == {"ticket": None, "address": None, "task": "", "model_name": "Room", "hidden_files": [],
                                                           "files": {"a.txt": "1"}, "run": score.run_part(spec)}


def test_the_image_tag_is_the_same_in_the_package_the_dockerfile_and_the_command():
    from fjallstation.image import fjallstation_task
    assert f"FJALLSTATION_IMAGE={image.IMAGE}\n" in (image.FOLDER / "Dockerfile").read_text() and f'"{image.IMAGE}"' in (image.FOLDER / "fjallstation_task.py").read_text()
    assert fjallstation_task.OMC_REWARD == score.OMC_REWARD


def test_the_command_builds_the_image_writes_the_example_and_starts_the_server(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(image.subprocess, "run", lambda command, **kw: calls.append(command))
    main(["image"])
    assert calls == [["docker", "build", "-t", image.IMAGE, str(image.FOLDER)]]
    main(["example", str(tmp_path / "store")])
    assert (tmp_path / "store/env/tasks-example/part-00000.parquet").exists()
    monkeypatch.setattr(app, "serve", lambda station, host, port: calls.append((len(station.tasks), host, port)) or type("S", (), {"serve_forever": lambda self: None})())
    main(["server", "--store", str(tmp_path / "store"), "--data", str(tmp_path / "rewards"), "--port", "8099"])
    assert calls[-1] == (5, "127.0.0.1", 8099)
    with pytest.raises(SystemExit):
        main([])
