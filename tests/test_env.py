import json
import threading
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fjallstation
from fjallstation import env as client
from fjallstation import score
from fjallstation import server as app

SPEC = {"model_name": "Room", "msl": "4.1.0", "libraries": ["Buildings 13.0.0"],
        "eval": {"start_time": 0, "stop_time": 10, "interval": 5, "tolerance": 1e-6, "outputs": ["T"]},
        "hidden_files": ["/opt/omlibrary/Buildings 13.0.0/Fluid/Examples/Room.mo", "/opt/omlibrary/Buildings 13.0.0/Resources/x y.txt"],
        "hidden_classes": ["Buildings.Fluid.Examples.Room"], "forbidden": ["Fluid.Examples.Room"]}
REF = {"time": [0.0, 5.0, 10.0], "T": [20.0, 21.0, 22.0]}


@pytest.fixture
def server(tmp_path):
    rows = [{"id": f"t-{i}", "prompt": f"Build room {i}.", "model_name": "Room", "spec": json.dumps(SPEC), "reference_code": "hidden",
             "reference_outputs": json.dumps(REF), "source_id": f"bldg-room-{i}"} for i in range(3)]
    (tmp_path / "store/env/tasks").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "store/env/tasks/part-00000.parquet")
    station = app.Station(app.load_tasks(tmp_path / "store"), tmp_path / "data", seed=0)
    httpd = app.serve(station, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path / "data"
    httpd.shutdown()


@pytest.fixture
def docker(monkeypatch):
    """A stand-in for docker: no container is started. `start` keeps the task file; `submit` answers as the image does."""
    calls, state = [], {}

    def run(command, input=None, **kw):
        calls.append(command)

        class Done:
            returncode, stdout, stderr = 0, "OpenModelica 1.27.0\n", ""
        if command[-2:] == ["fjallstation-task", "start"]:
            state["task"] = json.loads(input)
        if command[-2:] == ["fjallstation-task", "submit"]:
            task, code = state["task"], input
            status = ("no model" if not code.strip() else "uses the source example" if "Fluid.Examples.Room" in code else
                      "did not load" if "broken" in code else "check failed" if "unchecked" in code else "simulated")
            values = {} if status != "simulated" else {"T": REF["T"] if "right" in code else [20.0, 25.0, 30.0]}
            Done.stdout = json.dumps({"ticket": task["ticket"], "address": task["address"], "task": task["task"], "image": state.get("image", task["image"]),
                                      "status": status, "omc_reward": score.OMC_REWARD[status], "values": values, "seconds": 1.0})
        return Done()

    monkeypatch.setattr(client.subprocess, "run", run)
    return calls, state


def test_a_task_a_container_and_the_reward_in_two_parts(server, docker):
    (url, data), (calls, state) = server, docker
    env = fjallstation.Env(url)
    task = env.reset()
    assert task.prompt.startswith("Build room") and task.model_name == "Room" and task.container.startswith("fjallstation-env-")
    run, start = calls[0], calls[1]
    assert run[:2] == ["docker", "run"] and run[run.index("--network") + 1] == "none" and "fjallstation-env:2" in run
    assert run[run.index("--user") + 1] == "1000:1000" and run[run.index("--pids-limit") + 1] == "512"       # the agent is never root
    assert start == ["docker", "exec", "-i", "-u", "0", task.container, "fjallstation-task", "start"]           # the image hides the files, as root
    task_file = state["task"]
    assert task_file["hidden_files"] == SPEC["hidden_files"] and task_file["address"] == url + "/v1/results" and len(task_file["ticket"]) == 24
    assert task_file["run"] == score.run_part(SPEC)
    assert env.exec("omc --version", timeout=90, max_output=5000).output.startswith("OpenModelica") and "-u" not in calls[2]
    assert calls[2][-6:] == ["-c", client.RUN, "fjallstation", "90", "omc --version", "5000"]       # time limit and output limit in the container
    result = env.step("model Room right end Room;")
    assert result == client.Result(reward=1.0, feedback="correct", omc=0.3, scorer=1.0)
    assert calls[3][-2:] == ["fjallstation-task", "submit"] and calls[3][3:5] == ["-u", "0"]
    assert calls[-1][:3] == ["docker", "rm", "-f"] and env.task is None and env.result_file is None       # the container is removed
    record = json.loads((data / "rewards.jsonl").read_text().splitlines()[0])
    assert record["reward"] == 1.0 and record["omc_reward"] == 0.3 and record["task"] == task.id and record["outputs"] == {"T": 0.0}
    assert record["source_id"].startswith("bldg-room") and record["container_seconds"] == 1.0 and record["server_milliseconds"] < 1000


def test_each_stage_has_its_reward_and_a_short_status(server, docker):
    url = server[0]
    env = fjallstation.Env(url)
    env.reset("t-1")
    assert env.step("model Room end Room;") == client.Result(0.3, "outputs differ", omc=0.3, scorer=0.0)       # it simulates
    env.reset("t-1")
    assert env.step("model Room unchecked end Room;") == client.Result(0.1, "check failed", omc=0.1, scorer=0.0)   # it loads
    env.reset("t-1")
    assert env.step("model Room broken end Room;") == client.Result(0.0, "did not load", omc=0.0, scorer=0.0)
    env.reset("t-1")
    assert env.step("model Room extends Buildings.Fluid.Examples.Room; end Room;").feedback == "uses the source example"
    env.reset("t-1")
    assert env.step("") == client.Result(0.0, "no model", omc=0.0, scorer=0.0)


def test_one_result_for_each_ticket_and_clear_errors(server, docker):
    url = server[0]
    env = fjallstation.Env(url)
    env.reset()
    env.step("model Room right end Room;")
    used = {"ticket": docker[1]["task"]["ticket"], "image": "fjallstation-env:2", "status": "simulated", "values": {"T": REF["T"]}}
    again = client.post(url + "/v1/results", used)                # the same result again: the same answer, and no second record
    with pytest.raises(RuntimeError, match="409.*needs fjallstation-env:2"):
        client.post(url + "/v1/results", {**used, "image": "fjallstation-env:1"})
    assert again["reward"] == 1.0 and again["status"] == "correct" and len((server[1] / "rewards.jsonl").read_text().splitlines()) == 1
    with pytest.raises(RuntimeError, match="409.*result already"):
        client.post(url + "/v1/results", {**used, "values": {"T": [0.0, 0.0, 0.0]}})
    with pytest.raises(RuntimeError, match="404"):
        client.post(url + "/v1/results", {**used, "ticket": "not-a-ticket"})
    with pytest.raises(RuntimeError, match="404"):
        env.reset("t-not-there")
    with pytest.raises(RuntimeError, match="reset"):
        env.step("model Room end Room;")
    with pytest.raises(RuntimeError, match="step"):
        env.send()
    with pytest.raises(ConnectionError):
        fjallstation.Env("http://127.0.0.1:9").reset()


def test_the_answers_of_the_server_hold_no_reference_and_no_error_number(server, docker):
    url = server[0]
    env = fjallstation.Env(url)
    task_file = env._post("/v1/tasks/sample", {})
    assert set(task_file) == {"ticket", "address", "task", "prompt", "model_name", "image", "hidden_files", "run"}
    assert "limit" not in task_file["run"] and "reference" not in json.dumps(task_file)
    done = client.post(task_file["address"], {"ticket": task_file["ticket"], "image": task_file["image"], "status": "simulated", "values": {"T": [20.0, 25.0, 30.0]}})
    assert done == {"reward": 0.0, "omc_reward": 0.3, "status": "outputs differ", "task": task_file["task"], "image": "fjallstation-env:2"}


def test_a_result_of_another_image_version_is_refused_and_does_not_use_up_the_ticket(server, docker):
    (url, data), (calls, state) = server, docker
    env = fjallstation.Env(url)
    env.reset()
    state["image"] = "fjallstation-env:1"
    with pytest.raises(RuntimeError, match="409.*needs fjallstation-env:2"):
        env.step("model Room right end Room;")
    assert not (data / "rewards.jsonl").exists() and env.result_file is not None
    env.result_file["image"] = "fjallstation-env:2"
    assert env.send().reward == 1.0


def test_a_fault_of_the_server_gives_no_reward_and_the_client_keeps_the_result(server, docker, monkeypatch):
    url, data = server
    env = fjallstation.Env(url)
    env.reset()
    real = score.score
    monkeypatch.setattr(score, "score", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError, match="500.*could not score"):
        env.step("model Room right end Room;")
    assert not (data / "rewards.jsonl").exists() and env.task is None       # the container is gone; the result file is kept
    pending, calls = env.result_file, len(docker[0])
    with monkeypatch.context() as blocked:
        def unexpected_request(*args):
            pytest.fail("A pending result must block new work before any request")
        blocked.setattr(env, "_post", unexpected_request)
        with pytest.raises(RuntimeError, match="pending.*send.*discard_result"):
            env.reset()
        with pytest.raises(RuntimeError, match="pending.*send.*discard_result"):
            env.step("model Room right end Room;")
    assert env.result_file is pending and len(docker[0]) == calls
    with pytest.raises(RuntimeError, match="500.*could not score"):
        env.send()
    assert env.result_file is pending
    monkeypatch.setattr(score, "score", real)
    assert env.send() == client.Result(1.0, "correct", omc=0.3, scorer=1.0)   # the same result, sent again
    assert env.result_file is None
    env.reset()
    env.close()


def test_explicit_discard_allows_a_new_episode_without_sending_the_old_result(server, docker, monkeypatch):
    url, data = server
    env = fjallstation.Env(url)
    env.reset()
    post = env._post

    def unavailable(*args):
        raise ConnectionError("delivery unavailable")

    monkeypatch.setattr(env, "_post", unavailable)
    with pytest.raises(ConnectionError):
        env.step("model Room right end Room;")
    old_ticket, calls = env.result_file["ticket"], len(docker[0])
    env.discard_result()
    env.discard_result()                                       # no pending result: a harmless no-op
    assert env.result_file is None and len(docker[0]) == calls
    assert not (data / "rewards.jsonl").exists()
    monkeypatch.setattr(env, "_post", post)
    env.reset()
    assert env.step("model Room right end Room;").reward == 1.0
    records = [json.loads(line) for line in (data / "rewards.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["ticket"] != old_ticket


def test_another_worker_can_finish_while_one_env_has_a_pending_result(server, docker, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    url, data = server
    waiting = fjallstation.Env(url)
    waiting.reset()
    post = waiting._post

    def unavailable(*args):
        raise ConnectionError("delivery unavailable for this worker")

    monkeypatch.setattr(waiting, "_post", unavailable)
    with pytest.raises(ConnectionError):
        waiting.step("model Room right end Room;")
    pending = waiting.result_file

    def other_worker():
        with fjallstation.Env(url) as other:
            other.reset()
            return other.step("model Room right end Room;")

    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(other_worker).result(timeout=10).reward == 1.0
    assert waiting.result_file is pending
    with pytest.raises(RuntimeError, match="pending"):
        waiting.reset()
    monkeypatch.setattr(waiting, "_post", post)
    assert waiting.send().reward == 1.0
    records = [json.loads(line) for line in (data / "rewards.jsonl").read_text().splitlines()]
    assert len(records) == 2 and len({r["ticket"] for r in records}) == 2


def test_retry_after_a_lost_reply_preserves_the_result_and_does_not_score_twice(server, docker, monkeypatch):
    url, data = server
    env = fjallstation.Env(url)
    env.reset()
    post = env._post

    def lose_reply(path, body):
        post(path, body)                                       # the server has already stored the result
        raise ConnectionError("reply lost")

    monkeypatch.setattr(env, "_post", lose_reply)
    with pytest.raises(ConnectionError):
        env.step("model Room right end Room;")
    pending = env.result_file
    with pytest.raises(RuntimeError, match="pending"):
        env.reset()
    assert env.result_file is pending
    monkeypatch.setattr(env, "_post", post)
    assert env.send().reward == 1.0 and env.result_file is None
    assert len((data / "rewards.jsonl").read_text().splitlines()) == 1


def test_context_cleanup_keeps_a_pending_result(server, docker, monkeypatch):
    env = fjallstation.Env(server[0])

    def unavailable(*args):
        raise ConnectionError("delivery unavailable")

    with pytest.raises(ConnectionError):
        with env:
            env.reset()
            monkeypatch.setattr(env, "_post", unavailable)
            env.step("model Room right end Room;")
    pending, calls = env.result_file, len(docker[0])
    assert pending is not None and env.task is None
    env.close()
    assert env.result_file is pending and len(docker[0]) == calls


def test_an_incomplete_server_reply_does_not_clear_the_pending_result(monkeypatch):
    env = fjallstation.Env()
    pending = {"ticket": "waiting", "status": "simulated", "values": {"T": REF["T"]}}
    env.result_file = pending
    monkeypatch.setattr(env, "_post", lambda *a: {"reward": 1.0, "omc_reward": 0.3})  # missing status
    with pytest.raises(KeyError):
        env.send()
    assert env.result_file is pending


def test_requests_that_are_not_json_or_too_large_are_refused(server):
    url = server[0]
    def status(data, headers):
        request = urllib.request.Request(url + "/v1/results", data, headers)
        try:
            return urllib.request.urlopen(request, timeout=10).status
        except urllib.error.HTTPError as e:
            return e.code
    assert status(b"not json", {}) == 400 and status(b"[1, 2]", {}) == 400
    assert status(b"{}", {"Content-Length": str(app.MAX_BODY_BYTES + 1)}) == 413


def test_the_client_takes_the_reward_parts_from_the_server_and_sends_to_its_own_server(server, docker):
    url = server[0]
    env = fjallstation.Env(url)
    env.reset()
    env.result_file = {"ticket": docker[1]["task"]["ticket"], "address": "http://127.0.0.1:9/elsewhere", "image": "fjallstation-env:2",
                       "status": "simulated", "omc_reward": 999, "values": {"T": [20.0, 25.0, 30.0]}}
    result = env.send()                                           # the address and the reward number of the file are not used
    assert result == client.Result(reward=0.3, feedback="outputs differ", omc=0.3, scorer=0.0)


def test_a_start_that_fails_leaves_no_container(server, monkeypatch):
    calls = []

    def run(command, input=None, **kw):
        calls.append(command)

        class Done:
            returncode, stdout, stderr = (1, "", "fjallstation-task: not allowed") if command[-2:] == ["fjallstation-task", "start"] else (0, "", "")
        return Done()

    monkeypatch.setattr(client.subprocess, "run", run)
    env = fjallstation.Env(server[0])
    with pytest.raises(RuntimeError, match="docker exec failed"):
        env.reset()
    assert calls[-1][:3] == ["docker", "rm", "-f"] and env.task is None


def test_a_long_output_and_a_time_limit_are_named(server, monkeypatch):
    def run(command, input=None, **kw):
        class Done:
            returncode, stdout, stderr = (124, "x" * 50, "") if "exec" in command and "bash" in command else (0, "", "")
        return Done()

    monkeypatch.setattr(client.subprocess, "run", run)
    env = fjallstation.Env(server[0])
    env.task = client.Task("t", "p", "M", "c")
    done = env.exec("yes", timeout=5, max_output=50)
    assert done.code == 124 and "did not end in 5 seconds" in done.output and "cut at 50 bytes" in done.output


def test_a_result_with_fields_of_a_wrong_type_is_not_valid_and_gives_no_server_fault(server, docker):
    url = server[0]
    for bad in ({"status": []}, {"status": "simulated", "values": 123}, {"status": "simulated", "values": {"T": "abc"}}):
        task_file = client.post(url + "/v1/tasks/sample", {})
        done = client.post(url + "/v1/results", {"ticket": task_file["ticket"], "image": "fjallstation-env:2", **bad})
        assert done["status"] == "result not valid" and done["reward"] == 0.0


def test_requests_with_a_length_that_is_not_valid_are_refused(server):
    import http.client
    host, port = server[0].removeprefix("http://").split(":")
    for length in ("-1", "abc"):
        connection = http.client.HTTPConnection(host, int(port), timeout=10)
        connection.putrequest("POST", "/v1/tasks/sample")
        connection.putheader("Content-Length", length)
        connection.endheaders()
        assert connection.getresponse().status == 400
        connection.close()


def test_a_task_set_with_a_fault_is_refused_with_the_task_and_the_reason(tmp_path):
    def store(name, **change):
        spec = {**SPEC, **{k: v for k, v in change.items() if k in ("eval", "limit", "libraries")}}
        row = {"id": "t-1", "prompt": "p", "model_name": change.get("model_name", "Room"), "spec": json.dumps(spec),
               "reference_outputs": json.dumps(change.get("reference", REF))}
        rows = [row, row] if change.get("twice") else [row]
        folder = tmp_path / name / "env/tasks"
        folder.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), folder / "part-00000.parquet")
        return tmp_path / name

    cases = {"same id": dict(twice=True), "not allowed in model_name": dict(model_name='Room); system("id"); checkModel(Room'),
             "no required output": dict(eval={**SPEC["eval"], "outputs": []}),
             "whole number of intervals": dict(eval={**SPEC["eval"], "interval": 3}),
             "do not cover": dict(reference={"time": [0.0, 5.0], "T": [20.0, 21.0]}),
             "count is not": dict(reference={"time": [0.0, 5.0, 10.0], "T": [20.0, 21.0]}),
             "are constant": dict(reference={"time": [0.0, 5.0, 10.0], "T": [20.0, 20.0, 20.0]}),
             "in order": dict(reference={"time": [0.0, 8.0, 3.0, 10.0], "T": [20.0, 21.0, 22.0, 23.0]}),
             "a time that is not a number": dict(reference={"time": [0.0, float("nan"), 10.0], "T": [20.0, 21.0, 22.0]}),
             "numbers for stop_time": dict(eval={**SPEC["eval"], "tolerance": True}),
             "limit is not a number above zero": dict(limit={}),
             "not allowed in libraries": dict(libraries=["bad library text"])}
    for i, (reason, change) in enumerate(cases.items()):
        with pytest.raises(ValueError, match=f"Task 't-1' in .*{reason}"):
            app.load_tasks(store(f"s{i}", **change))
    assert len(app.load_tasks(store("good"))) == 1


@pytest.mark.parametrize("files, reason", [
    (["not-a-file-map"], "files must map"), ([], "files must map"), (None, "files must map"), (7, "files must map"),
    ({"../outside.txt": "text"}, "not a plain file name"), ({"/outside.txt": "text"}, "not a plain file name"),
    ({".hidden": "text"}, "not a plain file name"), ({"": "text"}, "not a plain file name"),
    ({"bad\0name": "text"}, "not a plain file name"), ({"data.txt": 7}, "must be text"),
])
def test_invalid_task_files_fail_loading_with_the_task_path_and_reason(tmp_path, files, reason):
    folder = tmp_path / "env/tasks"
    folder.mkdir(parents=True)
    path = folder / "part-00000.parquet"
    row = {"id": "t-files", "prompt": "p", "model_name": "Room", "spec": json.dumps(SPEC),
           "reference_outputs": json.dumps(REF), "files": json.dumps(files)}
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(ValueError) as fault:
        app.load_tasks(tmp_path)
    assert "Task 't-files'" in str(fault.value) and str(path) in str(fault.value) and reason in str(fault.value)


@pytest.mark.parametrize("files", [{}, {"weather.csv": "7", "empty.txt": "", "mätning.txt": "värme\n"}])
def test_task_file_text_reaches_the_worker_unchanged(tmp_path, files):
    from fjallstation.image import fjallstation_task as command

    folder = tmp_path / "store/env/tasks"
    folder.mkdir(parents=True)
    row = {"id": "t-files", "prompt": "p", "model_name": "Room", "spec": json.dumps({**SPEC, "hidden_files": []}),
           "reference_outputs": json.dumps(REF), "files": json.dumps(files)}
    pq.write_table(pa.Table.from_pylist([row]), folder / "part-00000.parquet")
    station = app.Station(app.load_tasks(tmp_path / "store"), tmp_path / "rewards")
    task = station.sample("t-files")
    assert task.get("files", {}) == files
    command.start(task, task_file=str(tmp_path / "task.json"), data_folder=str(tmp_path / "data"))
    assert {name: (tmp_path / "data" / name).read_text(encoding="utf-8") for name in files} == files


def test_the_server_keeps_a_bounded_number_of_tickets(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MAX_TICKETS", 3)
    rows = [{"id": "t-1", "prompt": "p", "model_name": "Room", "spec": json.dumps(SPEC), "reference_outputs": json.dumps(REF)}]
    (tmp_path / "s/env/tasks").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "s/env/tasks/part-00000.parquet")
    station = app.Station(app.load_tasks(tmp_path / "s"), tmp_path / "d")
    tickets = [station.sample()["ticket"] for _ in range(3)]
    with pytest.raises(app.StationFull, match="busy"):
        station.sample()
    assert list(station.tickets) == tickets
    station.result({"ticket": tickets[1], "image": station.image, "status": "simulated", "values": {"T": REF["T"]}})
    new = station.sample()["ticket"]
    assert list(station.tickets) == [tickets[0], tickets[2], new]


def test_a_full_server_returns_busy_and_existing_workers_can_finish(server, docker, monkeypatch):
    monkeypatch.setattr(app, "MAX_TICKETS", 1)
    first, other = fjallstation.Env(server[0]), fjallstation.Env(server[0])
    first.reset()
    calls = len(docker[0])
    with pytest.raises(RuntimeError, match="503.*busy"):
        other.reset()
    assert other.task is None and len(docker[0]) == calls
    assert first.step("model Room right end Room;").reward == 1.0
    other.reset()
    assert other.step("model Room right end Room;").reward == 1.0


def test_the_shell_line_of_a_command_keeps_the_exit_code_cuts_the_output_and_stops_at_the_limit():
    import subprocess
    import time

    def run(command, seconds=20, limit=100):
        started = time.monotonic()
        done = subprocess.run(["bash", "-c", client.RUN, "fjallstation", str(seconds), command, str(limit)], capture_output=True, timeout=30)
        return done.returncode, done.stdout, done.stderr, time.monotonic() - started

    assert run("echo hello; echo problem >&2; exit 3")[:3] == (3, b"hello\nproblem\n", b"")
    code, out, _, _ = run("for i in $(seq 1 20000); do echo x; done; exit 42")       # a noisy command runs to its end
    assert code == 42 and len(out) == 100
    code, _, _, seconds = run("sleep 30", seconds=1)
    assert code == 124 and seconds < 10


def test_output_of_a_command_that_is_not_text_does_not_stop_the_client(server, monkeypatch):
    real = client.subprocess.run
    monkeypatch.setattr(client.subprocess, "run", lambda command, **kw: real(["printf", r"caf\303\251 \377\376 end"], **{k: v for k, v in kw.items() if k != "input"}))
    env = fjallstation.Env(server[0])
    env.task = client.Task("t", "p", "M", "c")
    assert env.exec("cat file.bin").output.startswith("café ") and env.exec("x").output.endswith(" end")


def test_a_machine_with_no_docker_or_with_no_task_image_gets_a_clear_message(server, monkeypatch):
    def no_docker(command, **kw):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(client.subprocess, "run", no_docker)
    with pytest.raises(RuntimeError, match="Docker was not found"):
        fjallstation.Env(server[0]).reset()

    def no_image(command, **kw):
        class Done:
            returncode, stdout, stderr = (125, "", "Unable to find image 'fjallstation-env:2' locally\npull access denied") if command[1] == "run" else (0, "", "")
        return Done()

    monkeypatch.setattr(client.subprocess, "run", no_image)
    with pytest.raises(RuntimeError, match="fjallstation image"):
        fjallstation.Env(server[0]).reset()
