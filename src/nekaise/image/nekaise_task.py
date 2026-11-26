#!/usr/bin/python3 -I
"""The two commands of the task image. With them the container works with any program; no other code is needed.

    nekaise-task start  [task file]      at the start of an episode, as root
    nekaise-task submit [model file]     at the end of an episode, as root; prints the result file

`start` reads the task file that the server gave (a path, or standard input). It removes the library files that
hold the answer of the task, puts the data files of the task into /work/data, and keeps the task file.

`submit` tries the model of the agent (a path, or standard input) with the settings of the task: load, check,
simulate. It prints one JSON object, the result file: the ticket and the address from the task file, the name of
this image, the status, the reward of the OpenModelica part, and the values of the required outputs at the fixed
time steps. The user's side sends this file to the address in it; the server answers with the scorer reward.

The agent works in the container as a normal user. It cannot change OpenModelica, this file or the kept task
file. This file runs as root with Python in isolated mode, so it reads no start-up file of the agent. The model
of the agent is compiled and simulated as another user that has no rights (`WORKER`), so a model that runs a
shell command cannot change this file or the files of root. Call `submit` after the agent has stopped, and take
the result from the output of the command. This file uses the standard library and numpy only.
"""

import csv
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

IMAGE = os.environ.get("NEKAISE_IMAGE", "nekaise-env:1")
TASK_FILE = "/etc/nekaise/task.json"
DATA_FOLDER = "/work/data"
LIBRARY_ROOT = "/opt/omlibrary"
TIME_LIMIT = 1800                        # seconds for the compilation and the simulation of one model
MAX_RESULT_BYTES = 200_000_000           # a larger result file is not read
MAX_LOG_BYTES = 5_000_000                # of a longer log only this first part is read
WORKER = 65534                           # user and group id for the compilation and the simulation ("nobody")
NAME = r"[A-Za-z_][A-Za-z0-9_]*"
VERSION = r"[0-9][0-9A-Za-z.+-]*"
# The reward of the OpenModelica part, by the status of the try: how far the model came.
OMC_REWARD = {"no model": 0.0, "uses the source example": 0.0, "did not load": 0.0,       # the model does not load
              "check failed": 0.1,                                                          # it loads
              "simulation failed": 0.2, "simulation stopped early": 0.2, "time limit": 0.2,  # it checks
              "simulated": 0.3}                                                             # it simulates to the stop time


def check(task: dict) -> None:
    """Raise ValueError when a field of the task file cannot go into the OpenModelica script as it is."""
    run = task.get("run") or {}
    ev = run.get("eval") or {}
    fields = [("model_name", task.get("model_name"), NAME), ("msl", run.get("msl"), VERSION)]
    fields += [("libraries", lib, f"{NAME} {VERSION}") for lib in run.get("libraries", [])]
    fields += [("hidden_classes", c, rf"{NAME}(\.{NAME})*") for c in run.get("hidden_classes", [])]
    fields += [("outputs", name, r"[A-Za-z_][A-Za-z0-9_.\[\],]*") for name in ev.get("outputs") or [None]]
    for field, value, pattern in fields:
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"The task file has a value that is not allowed in {field}: {value!r}")
    try:
        start_time, stop, interval, tolerance = (float(ev.get(k, d)) for k, d in (("start_time", 0), ("stop_time", "x"), ("interval", "x"), ("tolerance", "x")))
    except (TypeError, ValueError):
        raise ValueError("The task file needs numbers for stop_time, interval and tolerance.") from None
    if not (all(math.isfinite(x) for x in (start_time, stop, interval, tolerance)) and interval > 0 and stop > start_time and tolerance > 0):
        raise ValueError("The task file needs a stop time after the start time, and an interval and a tolerance above zero.")


def hide(files: list, library_root: str = LIBRARY_ROOT) -> int:
    """Remove library files and their lines in `package.order`. Returns the number of removed files."""
    removed = 0
    for path in files:
        path = os.path.normpath(path)
        if not path.startswith(library_root.rstrip("/") + "/"):
            raise ValueError(f"Not a file of the library folder: {path}")
        if os.path.lexists(path):
            os.remove(path)
            removed += 1
        if path.endswith(".mo"):
            name, order = os.path.basename(path)[:-3], os.path.join(os.path.dirname(path), "package.order")
            if os.path.isfile(order):
                with open(order, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                with open(order, "w", encoding="utf-8") as f:
                    f.write("".join(line + "\n" for line in lines if line != name))
    return removed


def start(task: dict, task_file: str = TASK_FILE, data_folder: str = DATA_FOLDER, library_root: str = LIBRARY_ROOT) -> dict:
    """Prepare the container for the task. Returns a short report."""
    check(task)
    removed = hide(task.get("hidden_files") or [], library_root)
    files = task.get("files") or {}
    if files:
        os.makedirs(data_folder, exist_ok=True)
        os.chmod(data_folder, 0o755)
    for name, text in files.items():
        if os.path.basename(name) != name or name.startswith("."):
            raise ValueError(f"Not a plain file name: {name}")
        with open(os.path.join(data_folder, name), "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(os.path.join(data_folder, name), 0o644)
    kept = {**task, "files": sorted(files)}                      # the data is in the data folder; the task file keeps the names
    os.makedirs(os.path.dirname(task_file), exist_ok=True)
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    os.chmod(task_file, 0o644)
    return {"task": task.get("task"), "hidden_files_removed": removed, "data_files": sorted(files)}


def script(task: dict, file_name: str = "model.mo") -> str:
    """The OpenModelica script that loads, checks and simulates the model of the task."""
    run, name = task["run"], task["model_name"]
    ev = run["eval"]
    n = int(round((ev["stop_time"] - ev.get("start_time", 0)) / ev["interval"]))
    loads = f'loadModel(Modelica, {{"{run["msl"]}"}}); getErrorString();\n'
    for lib in run["libraries"]:
        lib_name, version = lib.split()
        loads += f'loadModel({lib_name}, {{"{version}"}}); getErrorString();\n'
    # The hidden classes go from this session too. A model then cannot reach the source example of the task
    # under another spelling, for example with `extends`.
    drop = "".join(f"deleteClass({c});\n" for c in run.get("hidden_classes", []))
    variables = "|".join(re.escape(v) for v in ev["outputs"])
    return (loads + f'loadFile("{file_name}"); getErrorString();\n' + drop
            + f"checkModel({name}); getErrorString();\n"
            + f'simulate({name}, startTime={ev.get("start_time", 0)}, stopTime={ev["stop_time"]}, numberOfIntervals={n}, '
              f'tolerance={ev["tolerance"]}, outputFormat="csv", variableFilter="{variables}", fileNamePrefix="out"); '
              "getErrorString();\n")


def read_regular(path: str, limit: int, head: bool = False):
    """The text of a regular file. None for a link, a pipe, a missing file, or a file above `limit` bytes.

    The file comes from a run of the submitted model, so it is opened with no wait and with no step through a link.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or (info.st_size > limit and not head):
            return None
        data = os.read(fd, limit) if head else b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
    finally:
        os.close(fd)
    return data.decode("utf-8", errors="replace")


def read_csv(text: str) -> dict:
    """The result table of a simulation: one list of numbers for each column, with the column `time`."""
    rows = list(csv.reader(text.splitlines()))
    names = [n.strip().strip('"') for n in rows[0]]
    table = {n: [] for n in names}
    for row in rows[1:]:
        if len(row) < len(names):
            continue
        for n, x in zip(names, row):
            try:
                table[n].append(float(x))
            except ValueError:
                table[n].append(math.nan)
    return table


def time_steps(ev: dict) -> list:
    """The fixed time steps of a task: from the start time to the stop time, one for each output interval."""
    n = int(round((ev["stop_time"] - ev.get("start_time", 0)) / ev["interval"]))
    return [ev.get("start_time", 0) + ev["interval"] * i for i in range(n + 1)]


def on_steps(table: dict, name: str, steps: list) -> list:
    """The values of one output at the fixed time steps.

    A simulation also writes rows at its events, two rows with the same time at each one. These extra rows are
    not used. At a time step that is also an event, the value after the event counts. A time step with no row
    gets the value between its two neighbours; that is rare.
    """
    import numpy as np
    t, y = np.asarray(table["time"], float), np.asarray(table[name], float)
    last = np.append(t[1:] != t[:-1], True)                       # the last row of each time
    return np.interp(np.asarray(steps, float), t[last], y[last]).tolist()


def read_run(log, result_path: str, task: dict) -> tuple:
    """The status and the table of a finished run, from its log and its result file."""
    run = task["run"]
    if log is None:
        return "simulation failed", None
    loads = re.findall(r"^(true|false)$", log, re.M)              # loadModel for each library, then loadFile
    n = 1 + len(run["libraries"])
    if len(loads) <= n or loads[n] != "true":
        return "did not load", None
    if f'Check of {task["model_name"]} completed successfully' not in log:
        return "check failed", None
    text = read_regular(result_path, MAX_RESULT_BYTES)            # None: no file, not a plain file, or too large
    if text is None:
        return "simulation failed", None
    try:
        table = read_csv(text)
    except IndexError:
        return "simulation failed", None
    if not table.get("time"):
        return "simulation failed", None
    ev = run["eval"]
    if not all(math.isfinite(t) for t in table["time"]):
        return "simulation failed", None
    if table["time"][-1] < ev["stop_time"] - 1e-6 * ev["interval"]:
        return "simulation stopped early", None
    return "simulated", table


def run_omc(folder: str, time_limit: int):
    """Run the script of `folder` with OpenModelica. Returns the log, or None at the time limit.

    As root, the run goes to the user `WORKER`, with the folder as its home. So the model has no rights of root,
    and it reads no OpenModelica settings that the agent left in its own home.
    """
    worker = {}
    if os.geteuid() == 0:
        for path in [folder] + [os.path.join(folder, name) for name in os.listdir(folder)]:
            os.chown(path, WORKER, WORKER)
        worker = {"user": WORKER, "group": WORKER, "extra_groups": []}
    with open(os.path.join(folder, "run.log"), "w") as log:
        try:
            done = subprocess.run(["timeout", "-k", "10", str(time_limit), "omc", "run.mos"], cwd=folder, stdout=log,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=time_limit + 60,
                                  env={**os.environ, "HOME": folder}, **worker)
        except subprocess.TimeoutExpired:
            return None
    if done.returncode in (124, 137):
        return None
    return read_regular(os.path.join(folder, "run.log"), MAX_LOG_BYTES, head=True) or ""


def submit(code: str, task: dict, run=run_omc) -> dict:
    """Try the model code of the agent for the task. Returns the result file as a dictionary."""
    started = time.time()
    result = {"ticket": task.get("ticket"), "address": task.get("address"), "task": task.get("task"), "image": IMAGE}

    def done(status: str, values: dict | None = None) -> dict:
        return {**result, "status": status, "omc_reward": OMC_REWARD[status], "values": values or {},
                "seconds": round(time.time() - started, 1)}

    if not code or not code.strip():
        return done("no model")
    if any(name in code for name in task["run"].get("forbidden", [])):
        return done("uses the source example")
    folder = tempfile.mkdtemp(prefix="nekaise-submit-")           # for this user only; the agent cannot reach it
    try:
        with open(os.path.join(folder, "model.mo"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(folder, "run.mos"), "w", encoding="utf-8") as f:
            f.write(script(task))
        log = run(folder, int(task["run"].get("time_limit", TIME_LIMIT)))
        if log is None:
            return done("time limit")
        status, table = read_run(log, os.path.join(folder, "out_res.csv"), task)
        if status != "simulated":
            return done(status)
        steps = time_steps(task["run"]["eval"])
        return done("simulated", {name: on_steps(table, name, steps) for name in task["run"]["eval"]["outputs"] if name in table})
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main(argv: list) -> int:
    if len(argv) < 2 or argv[1] not in ("start", "submit"):
        print(__doc__.strip().split("\n\n")[0] + "\n\n    nekaise-task start [task file]\n    nekaise-task submit [model file]", file=sys.stderr)
        return 2
    source = argv[2] if len(argv) > 2 and argv[2] != "-" else None
    text = open(source, encoding="utf-8", errors="replace").read() if source else sys.stdin.read()
    if os.geteuid() != 0:
        print("nekaise-task: run this command as root (for example: docker exec -u 0 ...).", file=sys.stderr)
        return 1
    if argv[1] == "start":
        try:
            print(json.dumps(start(json.loads(text))))
        except ValueError as fault:                              # not JSON, or a value that is not allowed
            print(f"nekaise-task: {fault}", file=sys.stderr)
            return 1
        return 0
    with open(TASK_FILE, encoding="utf-8") as f:
        task = json.load(f)
    print(json.dumps(submit(text, task)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
