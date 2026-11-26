import json

import pyarrow.parquet as pq
import pytest

from nekaise import example, image, score
from nekaise import server as app
from nekaise.__main__ import main


def simulated(code, spec, files=None, **kw):
    """In place of the task image: a rising line for each output."""
    steps = score.time_steps(spec["eval"])
    return {"status": "simulated", "omc_reward": 0.3, "values": {name: (20 + steps / 1000).tolist() for name in spec["eval"]["outputs"]}}


def test_the_example_store_has_the_three_parts_and_the_server_can_read_it(tmp_path):
    store = example.make(tmp_path / "store", try_model=simulated)
    assert len(pq.read_table(store / "corpus/example/part-00000.parquet")) == 3
    assert pq.read_table(store / "instruct/example/part-00000.parquet").to_pylist()[0]["messages"][0]["role"] == "user"
    rows = pq.read_table(store / "env/tasks-example/part-00000.parquet").to_pylist()
    assert [r["model_name"] for r in rows] == ["HeatedRoom", "WaterTank"] and "Write the complete Modelica model `HeatedRoom`" in rows[0]["prompt"]
    spec, reference = json.loads(rows[1]["spec"]), json.loads(rows[1]["reference_outputs"])
    assert spec["eval"]["outputs"] == ["T_tank"] and len(reference["time"]) == len(reference["T_tank"]) == 145 and reference["time"][-1] == 43200
    tasks = app.load_tasks(store)
    assert set(tasks) == {"example-heatedroom", "example-watertank"}
    station = app.Station(tasks, tmp_path / "rewards", seed=0)
    task_file = station.sample("example-heatedroom", "http://here/v1/results")
    assert task_file["image"] == image.IMAGE and task_file["run"]["libraries"] == [] and task_file["hidden_files"] == []
    right = {"ticket": task_file["ticket"], "image": image.IMAGE, "status": "simulated", "values": {"T_room": tasks["example-heatedroom"]["reference_outputs"]["T_room"]}}
    assert station.result(right)["reward"] == 1.0


def test_a_reference_model_that_does_not_simulate_stops_the_example(tmp_path):
    with pytest.raises(RuntimeError, match="HeatedRoom did not simulate"):
        example.make(tmp_path / "store", try_model=lambda code, spec, **kw: {"status": "check failed"})


def test_a_task_file_for_one_try_has_no_ticket_and_no_reference():
    spec = {"model_name": "Room", "msl": "4.1.0", "libraries": [], "eval": {"stop_time": 10, "interval": 5, "tolerance": 1e-6, "outputs": ["T"]}}
    assert image.task_file(spec, files={"a.txt": "1"}) == {"ticket": None, "address": None, "task": "", "model_name": "Room", "hidden_files": [],
                                                           "files": {"a.txt": "1"}, "run": score.run_part(spec)}


def test_the_image_tag_is_the_same_in_the_package_the_dockerfile_and_the_command():
    from nekaise.image import nekaise_task
    assert f"NEKAISE_IMAGE={image.IMAGE}\n" in (image.FOLDER / "Dockerfile").read_text() and f'"{image.IMAGE}"' in (image.FOLDER / "nekaise_task.py").read_text()
    assert nekaise_task.OMC_REWARD == score.OMC_REWARD


def test_the_command_builds_the_image_writes_the_example_and_starts_the_server(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(image.subprocess, "run", lambda command, **kw: calls.append(command))
    main(["image"])
    assert calls == [["docker", "build", "-t", image.IMAGE, str(image.FOLDER)]]
    monkeypatch.setattr(example.image, "try_model", simulated)
    main(["example", str(tmp_path / "store")])
    assert (tmp_path / "store/env/tasks-example/part-00000.parquet").exists()
    monkeypatch.setattr(app, "serve", lambda station, host, port: calls.append((len(station.tasks), host, port)) or type("S", (), {"serve_forever": lambda self: None})())
    main(["server", "--store", str(tmp_path / "store"), "--data", str(tmp_path / "rewards"), "--port", "8099"])
    assert calls[-1] == (2, "127.0.0.1", 8099)
    with pytest.raises(SystemExit):
        main([])
