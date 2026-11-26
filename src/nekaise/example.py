"""A small example store, to try the tool kit without other data.

    nekaise example ./store

It writes three text rows, two chats and two Env tasks in the format of docs/data.md. The reference outputs of
the tasks come from a simulation of the reference models in the task image, so build the image first
(`nekaise image`). Read this file as the pattern for a task set of your own.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nekaise import image

MSL = "4.1.0"
FOOTER = "\n\nWrite the complete Modelica model `{name}` in one file. Environment: Modelica Standard Library {msl}."

TEXTS = [
    "A heat pump moves heat from a cold source to a warm sink. Its coefficient of performance (COP) is the heat "
    "output divided by the electric power. The COP falls when the temperature difference between the sink and the source grows.",
    "A room loses heat to the outdoor air through its walls, its windows and its ventilation. In a steady state, the "
    "heat loss is the heat loss coefficient in W/K multiplied by the difference between the room and the outdoor temperature.",
    "The thermal mass of a building stores heat. A heavy building follows a change of the outdoor temperature slowly. "
    "Its time constant is the heat capacity divided by the heat loss coefficient.",
]
CHATS = [
    [{"role": "user", "content": "A room needs 1,200 W of heat. The heat pump has a COP of 3. How much electric power does it use?"},
     {"role": "assistant", "content": "The electric power is the heat divided by the COP: 1,200 W / 3 = 400 W."}],
    [{"role": "user", "content": "A room has a heat capacity of 2 MJ/K and a heat loss coefficient of 100 W/K. What is its time constant?"},
     {"role": "assistant", "content": "The time constant is the heat capacity divided by the heat loss coefficient: 2,000,000 J/K / 100 W/K = 20,000 s, which is about 5.6 hours."}],
]
TASKS = [
    {"name": "HeatedRoom", "outputs": ["T_room"], "stop_time": 86400, "interval": 600,
     "prompt": "A room is one heat capacity of 2,000,000 J/K. It loses heat to the outdoor air through a conductance of 100 W/K. The outdoor "
               "air has a constant temperature of 0 degC. A heater gives a constant heat flow of 1,500 W. At the start the room has 10 degC.\n\n"
               "Required output (a top-level variable of type Real): `T_room`, the room temperature in degC. The model is simulated from "
               "0 s to 86,400 s, and the output is compared at each 600 s.",
     "reference": "model HeatedRoom\n  parameter Real C = 2e6;\n  parameter Real UA = 100;\n  Real T(start = 10, fixed = true);\n"
                  "  output Real T_room;\nequation\n  C*der(T) = 1500 - UA*(T - 0);\n  T_room = T;\nend HeatedRoom;\n"},
    {"name": "WaterTank", "outputs": ["T_tank"], "stop_time": 43200, "interval": 300,
     "prompt": "A tank holds 200 kg of water with a specific heat capacity of 4,186 J/(kg K). It loses heat to a room of 20 degC through a "
               "conductance of 3 W/K. An electric heater of 2,000 W is on from 7,200 s to 10,800 s, and off at all other times. At the "
               "start the water has 60 degC.\n\nRequired output (a top-level variable of type Real): `T_tank`, the water temperature in "
               "degC. The model is simulated from 0 s to 43,200 s, and the output is compared at each 300 s.",
     "reference": "model WaterTank\n  parameter Real C = 200*4186;\n  parameter Real UA = 3;\n  Real T(start = 60, fixed = true);\n  Real Q;\n"
                  "  output Real T_tank;\nequation\n  Q = if time >= 7200 and time < 10800 then 2000 else 0;\n  C*der(T) = Q - UA*(T - 20);\n"
                  "  T_tank = T;\nend WaterTank;\n"},
]


def write(folder: Path, rows: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), folder / "part-00000.parquet")


def task_row(task: dict, try_model=None) -> dict:
    """One row of a task set: the reference model is simulated in the task image, and its output values are kept."""
    try_model = try_model or image.try_model
    spec = {"model_name": task["name"], "msl": MSL, "libraries": [], "hidden_files": [],
            "eval": {"start_time": 0, "stop_time": task["stop_time"], "interval": task["interval"], "tolerance": 1e-6, "outputs": task["outputs"]}}
    result = try_model(task["reference"], spec)
    if result.get("status") != "simulated":
        raise RuntimeError(f"The reference model {task['name']} did not simulate in the task image: {result.get('status')}")
    steps = [task["interval"] * i for i in range(task["stop_time"] // task["interval"] + 1)]
    return {"id": f"example-{task['name'].lower()}", "prompt": task["prompt"] + FOOTER.format(name=task["name"], msl=MSL),
            "model_name": task["name"], "spec": json.dumps(spec), "reference_outputs": json.dumps({"time": steps, **result["values"]}),
            "source_id": f"example/{task['name']}"}


def make(store: Path, try_model=None) -> Path:
    """Write the example store into the folder `store`."""
    store = Path(store)
    write(store / "corpus" / "example", [{"id": f"example-text-{i}", "text": text, "source": "example"} for i, text in enumerate(TEXTS, 1)])
    write(store / "instruct" / "example", [{"id": f"example-chat-{i}", "messages": chat} for i, chat in enumerate(CHATS, 1)])
    write(store / "env" / "tasks-example", [task_row(task, try_model) for task in TASKS])
    return store
