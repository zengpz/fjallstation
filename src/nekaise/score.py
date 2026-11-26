"""Score a result: compare the output values of a model with the reference outputs of its task.

The server runs no simulation. The container of the user simulates the model with the command `nekaise-task
submit` of the task image and sends the result file: the status of the try and the values of the required
outputs at the fixed time steps. The server compares these values with the reference outputs of the task.

The reward has two parts. The OpenModelica part says how far the model came in the container: 0 when it does
not load, 0.1 when it loads, 0.2 when it checks, 0.3 when it simulates to the stop time. The scorer part is 1
when every required output has an NRMSE at or below the limit against the reference (RMSE divided by the range
of the reference output), else 0. The limit is 1% for a task whose reference comes from a simulation. A task
with measured data as the reference names its own limit for each output; a time step with no measured value
is not compared.

A client that sends false values harms only its own training. So the result is not checked against the model
code. The reference outputs and the error of each output stay on the server.
"""

import math

import numpy as np

LIMIT = 0.01            # largest NRMSE of a correct output, for a task that names no limit of its own
EPS = 1e-12             # smallest range used as the divisor, for an output that is almost constant
# The reward of the OpenModelica part, by the status in the result file. The same table is in the image
# (image/nekaise_task.py); the server uses its own copy and not the number in the result file.
OMC_REWARD = {"no model": 0.0, "uses the source example": 0.0, "did not load": 0.0,
              "check failed": 0.1,
              "simulation failed": 0.2, "simulation stopped early": 0.2, "time limit": 0.2,
              "simulated": 0.3}


def run_part(spec: dict) -> dict:
    """The part of a task file that the submit command of the image needs to try a model."""
    return {"msl": spec["msl"], "libraries": spec["libraries"], "eval": spec["eval"],
            "hidden_classes": spec.get("hidden_classes", []), "forbidden": spec.get("forbidden", [])}


def time_steps(ev: dict) -> np.ndarray:
    """The fixed time steps of a task: from the start time to the stop time, one for each output interval."""
    n = int(round((ev["stop_time"] - ev.get("start_time", 0)) / ev["interval"]))
    return ev.get("start_time", 0) + ev["interval"] * np.arange(n + 1)


def on_steps(table: dict, name: str, steps: np.ndarray) -> np.ndarray:
    """The values of one output of a result table at the fixed time steps.

    A table of a simulation also has rows at its events, two rows with the same time at each one. These extra
    rows are not compared. At a time step that is also an event, the value after the event counts. A time step
    with no row gets the value between its two neighbours; that is rare. The image has the same function; here
    it reads the reference outputs of a task set that holds whole tables.
    """
    t, y = np.asarray(table["time"], float), np.asarray(table[name], float)
    last = np.append(t[1:] != t[:-1], True)                       # the last row of each time
    return np.interp(steps, t[last], y[last])


def nrmse(values: np.ndarray, reference: np.ndarray) -> float:
    span = max(float(np.max(reference) - np.min(reference)), EPS)
    return float(np.sqrt(np.mean((values - reference) ** 2)) / span)


def compare(values: dict, reference: dict, spec: dict, limit: float | dict | None = None) -> dict:
    """The scorer reward of output values at the fixed time steps, with the error of each output.

    `values` and `reference` hold one list for each output, with one value for each time step. The limit is one
    number, or a number for each output by name. Without `limit`, the limit of the task counts (`spec["limit"]`),
    and 1% when the task names none.
    """
    ev = spec["eval"]
    if limit is None:
        limit = spec.get("limit", LIMIT)
    limit_of = (lambda name: limit[name]) if isinstance(limit, dict) else (lambda name: limit)
    count, errors = len(time_steps(ev)), {}
    for name in ev["outputs"]:
        if name not in (values or {}):
            errors[name] = None
            continue
        try:
            g = np.asarray(values[name], float)
        except (TypeError, ValueError):
            return {"reward": 0.0, "status": "result not valid", "outputs": {}}
        if g.shape != (count,):
            return {"reward": 0.0, "status": "result not valid", "outputs": {}}
        r = np.asarray(reference[name], float)
        ok = np.isfinite(r)                                       # a time step with no measured value is not compared
        errors[name] = nrmse(g[ok], r[ok]) if ok.any() and np.all(np.isfinite(g[ok])) else math.inf
    if any(e is None for e in errors.values()):
        status = "output missing"
    elif all(e <= limit_of(name) for name, e in errors.items()):
        status = "correct"
    else:
        status = "outputs differ"
    return {"reward": 1.0 if status == "correct" else 0.0, "status": status, "outputs": errors}


def score(result: dict, spec: dict, reference: dict) -> dict:
    """The two parts of the reward for a result file, with the status and the error of each output."""
    status = result.get("status")
    if not isinstance(status, str) or status not in OMC_REWARD:
        return {"reward": 0.0, "omc_reward": 0.0, "status": "result not valid", "outputs": {}}
    if status != "simulated":                                    # the model did not come to the comparison
        return {"reward": 0.0, "omc_reward": OMC_REWARD[status], "status": status, "outputs": {}}
    if not isinstance(result.get("values"), dict):
        return {"reward": 0.0, "omc_reward": OMC_REWARD["simulated"], "status": "result not valid", "outputs": {}}
    return {**compare(result["values"], reference, spec), "omc_reward": OMC_REWARD["simulated"]}
