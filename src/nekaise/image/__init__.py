"""The task image: OpenModelica, the Modelica libraries of the tasks and the command `nekaise-task`.

    nekaise image                      builds the image from the files of this folder

`try_model` runs one model in a fresh container, with the same two commands that an episode uses. Use it to
make the reference outputs of a task of your own (see `nekaise/example.py`).
"""

import json
import secrets
import subprocess
import tempfile
from pathlib import Path

from nekaise import score

IMAGE = "nekaise-env:1"                  # the tag of the image; a result file names it
FOLDER = Path(__file__).parent
TIME_LIMIT = 1800                        # seconds for the compilation and the simulation of one model


def build() -> None:
    """Build the task image with Docker. The first build takes some minutes."""
    subprocess.run(["docker", "build", "-t", IMAGE, str(FOLDER)], check=True)


def task_file(spec: dict, name: str = "", files: dict | None = None) -> dict:
    """A task file for the commands of the image, without a ticket and an address."""
    return {"ticket": None, "address": None, "task": name, "model_name": spec["model_name"],
            "hidden_files": spec.get("hidden_files", []), "files": files or {}, "run": score.run_part(spec)}


def try_model(code: str, spec: dict, files: dict | None = None, image: str = IMAGE, cpus: float = 2, memory: str = "8g") -> dict:
    """The result file of `code` for the task of `spec`: {"status", "omc_reward", "values", ...}.

    Raises RuntimeError when the container cannot start or gives no result file; that is not a result of the model.
    """
    with tempfile.TemporaryDirectory(prefix="nekaise-try-") as folder:
        Path(folder, "task.json").write_text(json.dumps(task_file(spec, files=files)), encoding="utf-8")
        Path(folder, "model.mo").write_text(code, encoding="utf-8")
        Path(folder).chmod(0o755)
        container = f"nekaise-try-{secrets.token_hex(6)}"
        command = ["docker", "run", "--rm", "--name", container, "--network", "none", "--cpus", str(cpus), "--memory", memory,
                   "--pids-limit", "512", "-v", f"{folder}:/input:ro", image, "sh", "-c",
                   "nekaise-task start /input/task.json > /dev/null && nekaise-task submit /input/model.mo"]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=TIME_LIMIT + 300)
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)       # also when the wait ended early
    try:
        return json.loads(done.stdout)
    except ValueError:
        raise RuntimeError(f"The task image gave no result: {done.stderr.strip()[-300:]}") from None
