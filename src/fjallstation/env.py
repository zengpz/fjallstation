"""The client side of Env: a task from the server, a container to work in, and the reward.

    env    = fjallstation.Env()
    task   = env.reset()               # task.prompt: the modelling task; a container is started for it
    env.exec("omc --version")          # the agent of the user works in the container (task.container)
    result = env.step(model_code)      # result.reward, result.omc, result.scorer, result.feedback

This class is a small client. It does four steps that any program can do:

1. It asks the server for a task and gets a task file (with a ticket and the address for the result).
2. It starts a container of the task image with no network, and runs `fjallstation-task start` in it. This command
   of the image removes the library files that hold the answer and puts the data files of the task into
   /work/data.
3. After the agent has worked, it runs `fjallstation-task submit` in the container. This command tries the model
   (load, check, simulate) and prints the result file: the status, the reward of the OpenModelica part, and the
   output values at the fixed time steps.
4. It sends the result file to the server. The server compares the values with the reference outputs and
   answers with the reward.

The agent is the user's: a language model with its harness. It works in the container as a normal user. The two
commands of the image run as root, and the submitted model is simulated as a third user with no rights. The
container has no network, a limit for the number of processes, and limits for CPU and memory.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_SERVER = "http://127.0.0.1:8040"
AGENT = "1000:1000"                      # user and group of the agent in the container; never root
MAX_OUTPUT = 1_000_000                   # most bytes of the output of one command that come back
# In the container: stop the command at the time limit, keep the first part of its output and drop the rest.
# The exit code is that of the command. This is a guard against an accident, not against an agent that works against it.
RUN = 'exec 2>/dev/null; timeout -k 10 "$1" bash -c "$2" 2>&1 | { head -c "$3"; cat > /dev/null; }; exit "${PIPESTATUS[0]}"'


@dataclass
class Task:
    id: str
    prompt: str
    model_name: str          # the name that the submitted model must have
    container: str           # the Docker container of this task; the work folder in it is /work


@dataclass
class Result:
    reward: float            # 1.0 for a correct model; else the OpenModelica part: 0, 0.1, 0.2 or 0.3
    feedback: str            # a short status, for example "correct", "outputs differ" or "check failed"
    omc: float = 0.0         # OpenModelica part, from the container: 0 does not load, 0.1 loads, 0.2 checks, 0.3 simulates
    scorer: float = 0.0      # scorer part, from the server: 1.0 when the outputs agree with the reference, else 0


@dataclass
class Run:
    code: int                # the exit code of the command
    output: str              # its output and error text


def _docker(*args: str, timeout: int | None = None, check: bool = True, input: str | None = None) -> subprocess.CompletedProcess:
    done = subprocess.run(["docker", *args], capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, input=input)
    if check and done.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {done.stderr.strip()[:500]}")
    return done


def post(address: str, body: dict, timeout: int = 600) -> dict:
    """Send one JSON object to the server and return its answer."""
    request = urllib.request.Request(address, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"The server refused the request ({e.code}): {e.read().decode(errors='replace')[:300]}") from None
    except urllib.error.URLError as e:
        raise ConnectionError(f"No answer from the server at {address}: {e.reason}. Start it with: fjallstation server --store <folder>") from None


class Env:
    def __init__(self, url: str | None = None, timeout: int = 3600, cpus: float = 2, memory: str = "8g"):
        self.url = (url or os.environ.get("FJALLSTATION_SERVER") or DEFAULT_SERVER).rstrip("/")
        self.timeout, self.cpus, self.memory = timeout, cpus, memory
        self.task: Task | None = None
        self.result_file: dict | None = None        # the result file of the last submission, until the server has it

    def _post(self, path: str, body: dict) -> dict:
        return post(self.url + path, body, self.timeout)

    def reset(self, task: str | None = None) -> Task:
        """A new task from the server and a fresh container for it. `task` asks for one task by its id."""
        self.close()
        task_file = self._post("/v1/tasks/sample", {"task": task} if task else {})
        container = f"fjallstation-env-{task_file['ticket'][:12]}"
        _docker("run", "-d", "--rm", "--name", container, "--network", "none", "--cpus", str(self.cpus), "--memory", self.memory,
                "--pids-limit", "512", "--user", AGENT, task_file["image"], "sleep", "infinity")
        self.task = Task(task_file["task"], task_file["prompt"], task_file["model_name"], container)
        try:
            _docker("exec", "-i", "-u", "0", container, "fjallstation-task", "start", input=json.dumps(task_file))
        except Exception:
            self.close()                                         # no container stays when the start fails
            raise
        return self.task

    def exec(self, command: str, timeout: int = 1800, max_output: int = MAX_OUTPUT) -> Run:
        """Run a shell command in the container of the task, in /work, as the normal user.

        The container stops the command after `timeout` seconds (exit code 124) and gives the first `max_output`
        bytes of its output. A process that the command leaves in the background must write to a file; else the
        command waits for it.
        """
        if self.task is None:
            raise RuntimeError("No task is open. Call reset() first.")
        try:
            done = _docker("exec", "-w", "/work", self.task.container, "bash", "-c", RUN, "fjallstation", str(int(timeout)), command,
                           str(int(max_output)), timeout=timeout + 60, check=False)
        except subprocess.TimeoutExpired:
            return Run(124, f"The command did not end in {timeout} seconds.")
        output = done.stdout + done.stderr
        if done.returncode == 124:
            output += f"\nThe command did not end in {timeout} seconds."
        if len(done.stdout.encode()) >= max_output:
            output += f"\nThe output was cut at {max_output} bytes."
        return Run(done.returncode, output)

    def step(self, model_code: str) -> Result:
        """Submit the model code for the open task. The container tries it; the server scores the result. The episode ends."""
        if self.task is None:
            raise RuntimeError("No task is open. Call reset() first.")
        try:
            done = _docker("exec", "-i", "-u", "0", self.task.container, "fjallstation-task", "submit", input=model_code, timeout=self.timeout)
            self.result_file = json.loads(done.stdout)
        finally:
            self.close()                                         # the result file holds all that the server needs
        return self.send()

    def send(self) -> Result:
        """Send the kept result file to the server. Call it again when the server did not answer."""
        if self.result_file is None:
            raise RuntimeError("No result to send. Call step() first.")
        answer = self._post("/v1/results", self.result_file)          # to the server of this client, not to an address from the container
        omc, scorer = float(answer["omc_reward"]), float(answer["reward"])
        self.result_file = None
        return Result(reward=1.0 if scorer == 1.0 else omc, feedback=answer["status"], omc=omc, scorer=scorer)

    def close(self) -> None:
        """Remove the container of the open task."""
        if self.task is not None:
            _docker("rm", "-f", self.task.container, check=False)
        self.task = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
