"""The two commands of the task image (nekaise/image/nekaise_task.py), tested without a container."""
import json
import os
from pathlib import Path

import pytest

from nekaise import image, score
from nekaise.image import nekaise_task as command

TASK = {"ticket": "abc123", "address": "http://server/v1/results", "task": "t-1", "model_name": "Room",
        "run": {"msl": "4.1.0", "libraries": ["Buildings 13.0.0"], "hidden_classes": ["Buildings.Fluid.Examples.Room"], "forbidden": ["Fluid.Examples.Room"],
                "eval": {"start_time": 0, "stop_time": 10, "interval": 5, "tolerance": 1e-6, "outputs": ["T", "Q"]}}}
OK = 'true\n""\ntrue\n""\ntrue\n""\n"Check of Room completed successfully."\n'


def test_the_script_loads_checks_and_simulates_with_the_settings_of_the_task():
    text = command.script(TASK)
    assert 'loadModel(Modelica, {"4.1.0"})' in text and 'loadModel(Buildings, {"13.0.0"})' in text
    assert text.index('loadFile("model.mo")') < text.index("deleteClass(Buildings.Fluid.Examples.Room)") < text.index("checkModel(Room)")
    assert "stopTime=10, numberOfIntervals=2, tolerance=1e-06" in text and 'variableFilter="T|Q"' in text


def test_start_removes_the_answer_files_and_puts_the_data_files_in_place(tmp_path):
    library = tmp_path / "omlibrary"
    examples = library / "Buildings 13.0.0/Fluid/Examples"
    examples.mkdir(parents=True)
    (examples / "Room.mo").write_text("model Room end Room;")
    (examples / "Other.mo").write_text("model Other end Other;")
    (examples / "package.order").write_text("Other\nRoom\nRoomWithWindow\n")
    results = library / "Buildings 13.0.0/Resources/x y.txt"
    results.parent.mkdir()
    results.write_text("data")
    task = TASK | {"hidden_files": [str(examples / "Room.mo"), str(results), str(examples / "NotThere.mo")],
                   "files": {"evaluation.txt": "#1\n"}, "prompt": "Build a room."}
    report = command.start(task, str(tmp_path / "etc/task.json"), str(tmp_path / "work/data"), str(library))
    assert report == {"task": "t-1", "hidden_files_removed": 2, "data_files": ["evaluation.txt"]}
    assert not (examples / "Room.mo").exists() and not results.exists() and (examples / "Other.mo").exists()
    assert (examples / "package.order").read_text() == "Other\nRoomWithWindow\n"            # only the line of the removed class goes
    assert (tmp_path / "work/data/evaluation.txt").read_text() == "#1\n"
    kept = json.loads((tmp_path / "etc/task.json").read_text())
    assert kept["ticket"] == "abc123" and kept["files"] == ["evaluation.txt"] and kept["run"] == TASK["run"]
    with pytest.raises(ValueError, match="library folder"):
        command.hide(["/etc/passwd"], str(library))
    with pytest.raises(ValueError, match="plain file name"):
        command.start(TASK | {"files": {"../x": ""}}, str(tmp_path / "etc/task.json"), str(tmp_path / "work/data"), str(library))


def test_the_log_gives_the_status(tmp_path):
    path = str(tmp_path / "out_res.csv")
    assert command.read_run('true\n""\ntrue\n""\nfalse\n"Error"\n', path, TASK) == ("did not load", None)
    assert command.read_run('true\n""\ntrue\n""\ntrue\n""\n"Error: Base class not found"\n', path, TASK) == ("check failed", None)
    assert command.read_run(OK, path, TASK) == ("simulation failed", None)
    Path(path).write_text('"time","T","Q"\n0,20,0\n5,25,500\n')
    assert command.read_run(OK, path, TASK) == ("simulation stopped early", None)          # the result must reach the stop time
    Path(path).write_text('"time","T","Q"\n0,20,0\n10,30,1000\n')
    assert command.read_run(OK, path, TASK) == ("simulated", {"time": [0.0, 10.0], "T": [20.0, 30.0], "Q": [0.0, 1000.0]})


def test_only_the_values_at_the_fixed_time_steps_go_into_the_result():
    ev = {"start_time": 0, "stop_time": 2, "interval": 1}
    assert command.time_steps(ev) == [0, 1, 2]
    table = {"time": [0, 1, 1.4, 1.4, 1.5, 1.5, 2], "y": [0, 0, 0, 0, 0, 1, 1]}              # rows at events are extra rows
    assert command.on_steps(table, "y", [0, 1, 2]) == [0.0, 0.0, 1.0]
    assert command.on_steps({"time": [0, 1, 1, 2], "y": [0, 0, 1, 1]}, "y", [0, 1, 2]) == [0.0, 1.0, 1.0]     # the value after an event counts
    assert command.on_steps(table, "y", [0, 1, 2]) == score.on_steps(table, "y", score.time_steps(ev)).tolist()   # as on the server


def test_submit_gives_the_result_file_with_the_ticket_the_status_and_the_values():
    def run(folder, time_limit):
        assert "checkModel(Room)" in open(os.path.join(folder, "run.mos")).read() and time_limit == 1800
        code = open(os.path.join(folder, "model.mo")).read()
        if "slow" in code:
            return None
        if "broken" in code:
            return 'true\n""\ntrue\n""\nfalse\n"Error"\n'
        open(os.path.join(folder, "out_res.csv"), "w").write('"time","T","Q"\n0,20,0\n5,25,500\n10,30,1000\n' if "both" in code else '"time","T"\n0,20\n5,25\n10,30\n')
        return OK

    good = command.submit("model Room both end Room;", TASK, run)
    assert {k: good[k] for k in ("ticket", "address", "task", "image", "status", "omc_reward")} == {
        "ticket": "abc123", "address": "http://server/v1/results", "task": "t-1", "image": command.IMAGE, "status": "simulated", "omc_reward": 0.3}
    assert good["values"] == {"T": [20.0, 25.0, 30.0], "Q": [0.0, 500.0, 1000.0]} and json.loads(json.dumps(good)) == good
    assert command.submit("model Room end Room;", TASK, run)["values"] == {"T": [20.0, 25.0, 30.0]}     # a missing output is left out
    assert [command.submit(code, TASK, run)[k] for code in ("model Room broken end Room;", "model Room slow end Room;") for k in ("status", "omc_reward")] == [
        "did not load", 0.0, "time limit", 0.2]
    never = lambda *a: (_ for _ in ()).throw(AssertionError("no run expected"))
    assert command.submit("  ", TASK, never)["status"] == "no model"
    used = command.submit("model Room extends Buildings.Fluid.Examples.Room; end Room;", TASK, never)
    assert used["status"] == "uses the source example" and used["omc_reward"] == 0.0 and used["values"] == {}


def test_the_reward_table_is_the_same_in_the_image_and_on_the_server():
    assert command.OMC_REWARD == score.OMC_REWARD
    assert sorted(set(command.OMC_REWARD.values())) == [0.0, 0.1, 0.2, 0.3]


def test_a_result_that_is_a_link_a_pipe_or_too_large_is_not_read(tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text('"time","T","Q"\n0,20,0\n10,30,1000\n')
    assert command.read_regular(str(secret), 1000).startswith('"time"')
    link = tmp_path / "link.csv"
    link.symlink_to(secret)
    assert command.read_regular(str(link), 1000) is None and command.read_run(OK, str(link), TASK) == ("simulation failed", None)
    pipe = tmp_path / "pipe.csv"
    os.mkfifo(pipe)
    assert command.read_regular(str(pipe), 1000) is None                    # and the call returns at once
    assert command.read_regular(str(secret), 10) is None and command.read_regular(str(secret), 10, head=True) == '"time","T"'
    assert command.read_regular(str(tmp_path / "missing.csv"), 1000) is None


def test_the_commands_need_root_and_name_their_use(capsys, monkeypatch):
    assert command.main(["nekaise-task"]) == 2 and "nekaise-task submit" in capsys.readouterr().err
    monkeypatch.setattr(command.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(command.sys, "stdin", type("In", (), {"read": staticmethod(lambda: "{}")})())
    assert command.main(["nekaise-task", "start"]) == 1 and "as root" in capsys.readouterr().err


def test_a_task_file_with_a_value_that_could_change_the_script_is_refused(tmp_path, capsys, monkeypatch):
    command.check(TASK)
    for change in ({"model_name": 'Room); system("id"); checkModel(Room'}, {"run": {**TASK["run"], "msl": '4.1.0"}); system("id");'}},
                   {"run": {**TASK["run"], "libraries": ["Buildings 13.0.0 x"]}}, {"run": {**TASK["run"], "hidden_classes": ['A"); system("id"']}},
                   {"run": {**TASK["run"], "eval": {**TASK["run"]["eval"], "outputs": ['T", fileNamePrefix="x']}}},
                   {"run": {**TASK["run"], "eval": {**TASK["run"]["eval"], "interval": 0}}}):
        with pytest.raises(ValueError):
            command.check({**TASK, **change})
    with pytest.raises(ValueError, match="model_name"):
        command.start({**TASK, "model_name": "a b"}, task_file=str(tmp_path / "task.json"), data_folder=str(tmp_path / "data"), library_root=str(tmp_path))
    assert not (tmp_path / "task.json").exists()


def test_a_time_column_with_a_value_that_is_not_a_number_is_a_failed_simulation(tmp_path):
    path = tmp_path / "out_res.csv"
    path.write_text('"time","T","Q"\n0,20,0\n5,25,500\nnan,30,1000\n')
    assert command.read_run(OK, str(path), TASK)[0] == "simulation failed"


def test_the_model_runs_as_a_user_without_rights_and_with_its_own_home(tmp_path, monkeypatch):
    seen = {}

    def run(args, **kw):
        seen.update(args=args, kw=kw)

        class Done:
            returncode = 0
        return Done()

    (tmp_path / "run.mos").write_text("x")
    monkeypatch.setattr(command.subprocess, "run", run)
    monkeypatch.setattr(command.os, "geteuid", lambda: 0)
    owners = []
    monkeypatch.setattr(command.os, "chown", lambda path, uid, gid: owners.append((os.path.basename(path), uid, gid)))
    command.run_omc(str(tmp_path), 60)
    assert seen["kw"]["user"] == seen["kw"]["group"] == command.WORKER == 65534 and seen["kw"]["extra_groups"] == [] and seen["kw"]["env"]["HOME"] == str(tmp_path)
    assert (tmp_path.name, 65534, 65534) in owners and ("run.mos", 65534, 65534) in owners
    assert (Path(command.__file__)).read_text().startswith("#!/usr/bin/python3 -I\n")          # isolated mode: no start-up file of the agent is read
