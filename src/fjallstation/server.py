"""The server side of Env: it gives tasks and it scores results. It runs no simulation.

    fjallstation server --store <store folder>          (or: python -m fjallstation.server --store <store folder>)

The server reads the task sets of the store: `env/tasks` and each set `env/tasks-<name>`. A client asks for a task
and gets a task file: the prompt, the name of the image, the library files to hide, the settings that the submit
command of the image needs, the data files of a task with measured data, a ticket for this task instance, and
the address for the result. The container of the client tries the model of its agent and makes a result file:
the status of the try and the output values at the fixed time steps. The client sends the result file to the
address in it, once for each ticket. The server compares the values with the reference outputs and answers with
the scorer reward and a short status. The reference outputs and the errors of the outputs stay on the server.

    POST /v1/tasks/sample   {}  or  {"task": "<task id>"}   -> the task file: {"ticket", "address", "task", "prompt",
                                                               "model_name", "image", "hidden_files", "run"[, "files"]}
    POST /v1/results        the result file                 -> {"reward", "omc_reward", "status", "task", "image"}
    GET  /v1/health                                          -> {"status": "ok", "tasks": <count>, "image"}
"""

import argparse
import glob
import hashlib
import json
import math
import random
import re
import secrets
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow.parquet as pq

from fjallstation import score
from fjallstation.image import IMAGE, fjallstation_task

MAX_BODY_BYTES = 50_000_000              # a longer request is refused
MAX_TICKETS = 200_000                    # active and completed records; only completed tickets can be removed


class StationFull(RuntimeError):
    """No slot for a new episode; all retained tickets are active."""


def check_task(row: dict, spec: dict, table: dict) -> None:
    """Raise ValueError when a task cannot be served: settings or reference values that are not complete or not clear."""
    if not all(isinstance(spec.get(key), list) for key in ("libraries", "hidden_files")) or not isinstance(spec.get("eval"), dict):
        raise ValueError("the settings need msl, libraries, hidden_files and eval")
    if not isinstance(spec["eval"].get("outputs"), list) or not spec["eval"]["outputs"]:
        raise ValueError("the settings name no required output")
    fjallstation_task.check({"model_name": row.get("model_name"), "run": score.run_part(spec)})     # the rules of the image command
    ev = spec["eval"]
    outputs, start, stop, interval = ev["outputs"], ev.get("start_time", 0), ev["stop_time"], ev["interval"]
    limit = spec.get("limit", score.LIMIT)
    limits = [limit.get(name) for name in outputs] if isinstance(limit, dict) else [limit]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in limits):
        raise ValueError("the limit is not a number above zero, or an output has no limit")
    time_ = table.get("time") if isinstance(table, dict) else None
    if not isinstance(time_, list) or not time_ or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in time_):
        raise ValueError("the reference values have no times, or a time that is not a number")
    if any(a > b for a, b in zip(time_, time_[1:])) or time_[0] > start + 1e-9 * interval or time_[-1] < stop - 1e-6 * interval:
        raise ValueError("the reference values do not cover the time from start_time to stop_time in order")
    for name in outputs:
        if not isinstance(table.get(name), list) or len(table[name]) != len(time_):
            raise ValueError(f"the reference values of {name} are missing, or their count is not the count of the times")


def load_tasks(store: Path) -> dict:
    """The tasks of the sets `env/tasks` and `env/tasks-<name>`, by task id."""
    tasks = {}
    for f in sorted(glob.glob(str(Path(store) / "env" / "tasks*" / "part-*.parquet"))):
        if not re.fullmatch(r"tasks(-[a-z0-9-]+)?", Path(f).parent.name):      # not a folder of a build in work
            continue
        for row in pq.read_table(f).to_pylist():
            try:
                spec, table = json.loads(row["spec"]), json.loads(row["reference_outputs"])
                check_task(row, spec, table)
                if row["id"] in tasks:
                    raise ValueError("another task has the same id")
                if row.get("files"):
                    spec["files"] = json.loads(row.pop("files"))           # the data files of a task with measured data
                steps = score.time_steps(spec["eval"])            # the scoring needs the fixed time steps only
                reference = {"time": steps.tolist()} | {n: score.on_steps(table, n, steps).tolist() for n in spec["eval"]["outputs"]}
                for name, values in reference.items():
                    finite = [v for v in values if math.isfinite(v)]
                    if name != "time" and (not finite or max(finite) - min(finite) <= 1e-9):
                        raise ValueError(f"the reference values of {name} are constant; the error is measured against their range")
            except (KeyError, TypeError, ValueError) as fault:
                raise ValueError(f"Task {row.get('id')!r} in {f}: {fault}") from None
            tasks[row["id"]] = {**row, "spec": spec, "reference_outputs": reference}
    if not tasks:
        raise FileNotFoundError(f"No task in {Path(store) / 'env'}. The server needs a set such as env/tasks-<name> (see docs/data.md).")
    return tasks


class Station:
    """The state of the server: the tasks, the open tickets and the record of the rewards."""

    def __init__(self, tasks: dict, data: Path, image: str = IMAGE, seed: int | None = None):
        self.tasks, self.image = tasks, image
        self.data = Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.random = random.Random(seed)
        self.tickets = {}
        self.completed = deque()        # tickets in completion order, for constant-time cleanup
        self.lock = threading.Lock()

    def sample(self, task_id: str | None = None, address: str = "") -> dict:
        """A task file for one new task instance."""
        with self.lock:
            if task_id is None:
                task_id = self.random.choice(sorted(self.tasks))
            if task_id not in self.tasks:
                raise KeyError(task_id)
            while len(self.tickets) >= MAX_TICKETS:
                if not self.completed:
                    raise StationFull(f"The server is busy: all {MAX_TICKETS} ticket slots are active. Retry when an episode finishes.")
                del self.tickets[self.completed.popleft()]
            ticket = secrets.token_hex(12)
            self.tickets[ticket] = {"task": task_id, "state": "open", "at": time.time()}
        task = self.tasks[task_id]
        answer = {"ticket": ticket, "address": address, "task": task_id, "prompt": task["prompt"],
                  "model_name": task["model_name"], "image": self.image, "hidden_files": task["spec"]["hidden_files"],
                  "run": score.run_part(task["spec"])}
        if task["spec"].get("files"):
            answer["files"] = task["spec"]["files"]
        return answer

    def result(self, result: dict) -> dict:
        """Score one result file. One result for each ticket; the same result again gets the same answer."""
        started = time.perf_counter()
        ticket = str(result.get("ticket"))
        mark = hashlib.sha256(json.dumps([result.get("status"), result.get("values")], sort_keys=True, default=str).encode()).hexdigest()
        with self.lock:
            entry = self.tickets.get(ticket)
            if entry is None:
                raise KeyError(ticket)
            if result.get("image") != self.image:                 # other versions can give other values
                raise PermissionError(f"The result comes from the image {result.get('image')}. This server needs {self.image}.")
            if entry["state"] == "done" and entry["mark"] == mark:
                return entry["answer"]                            # the answer got lost on its way; the client sent the result again
            if entry["state"] != "open":
                raise PermissionError("This ticket has its result already.")
            entry["state"] = "scoring"
        task = self.tasks[entry["task"]]
        try:
            scored = score.score(result, task["spec"], task["reference_outputs"])
            record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ticket": ticket, "task": entry["task"],
                      "source_id": task.get("source_id"), "reward": scored["reward"], "omc_reward": scored["omc_reward"],
                      "status": scored["status"], "outputs": scored["outputs"], "image": self.image,
                      "container_seconds": result.get("seconds"), "server_milliseconds": round(1000 * (time.perf_counter() - started), 2)}
            answer = {"reward": scored["reward"], "omc_reward": scored["omc_reward"], "status": scored["status"],
                      "task": entry["task"], "image": self.image}
            with self.lock:
                with (self.data / "rewards.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                entry.update(state="done", mark=mark, answer=answer)
                self.completed.append(ticket)
        except Exception:
            with self.lock:
                entry["state"] = "open"                           # a fault of the server does not use up the ticket
            raise
        return answer


def handler(station: Station):
    class Handler(BaseHTTPRequestHandler):
        def answer(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/health":
                self.answer(200, {"status": "ok", "tasks": len(station.tasks), "image": station.image})
            else:
                self.answer(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                return self.answer(400, {"error": "The request has no valid length."})
            if length > MAX_BODY_BYTES:
                return self.answer(413, {"error": "The request is too large."})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self.answer(400, {"error": "The request body is not JSON."})
            if not isinstance(body, dict):
                return self.answer(400, {"error": "The request body is not a JSON object."})
            try:
                if self.path == "/v1/tasks/sample":
                    address = f"http://{self.headers.get('Host') or '%s:%s' % self.server.server_address[:2]}/v1/results"
                    return self.answer(200, station.sample(body.get("task"), address))
                if self.path == "/v1/results":
                    return self.answer(200, station.result(body))
            except KeyError as e:
                return self.answer(404, {"error": f"Not known: {e.args[0]}"})
            except PermissionError as e:
                return self.answer(409, {"error": str(e)})
            except StationFull as e:
                return self.answer(503, {"error": str(e)})
            except Exception as e:                                # no reward and no record for a fault of the server
                return self.answer(500, {"error": f"The server could not score the result: {e}"})
            self.answer(404, {"error": "not found"})

        def log_message(self, *args):                             # the reward record is the log
            pass

    return Handler


def serve(station: Station, host: str = "127.0.0.1", port: int = 8040) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler(station))


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fjallstation server", description=__doc__.split("\n")[0])
    parser.add_argument("--store", type=Path, default=Path("store"), help="the folder with the data; the server reads its env/ part (default: ./store)")
    parser.add_argument("--data", type=Path, default=Path("rewards"), help="folder for the reward record (default: ./rewards)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--image", default=IMAGE, help="the image version that the clients must use")
    parser.add_argument("--seed", type=int, help="fixes the order of the sampled tasks")
    args = parser.parse_args(argv)
    station = Station(load_tasks(args.store), args.data, args.image, args.seed)
    server = serve(station, args.host, args.port)
    print(f"fjallstation server: {len(station.tasks)} tasks, image {station.image}, http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
