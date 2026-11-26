# The data folder

All data is in one folder, the store. `FJALLSTATION_STORE` names it for `Corpus()` and `Instruct()`; `fjallstation server --store` names it for the server. The default of both is `./store`.

```
store/
  corpus/<set>/part-*.parquet       text
  instruct/<set>/part-*.parquet     chats
  env/tasks-<name>/part-*.parquet   tasks of Env
```

A set is a folder with one or more Parquet part files. You can have many sets and any of the three parts. `fjallstation example ./store` writes a small store of this form; [`example.py`](https://github.com/zengpz/fjallstation/blob/main/src/fjallstation/example.py) is the pattern for your own.

## Text: `corpus`

| Column | Content |
|---|---|
| `id` | a name for the row |
| `text` | the text |
| `source` | where the text comes from, in one word |

## Chats: `instruct`

| Column | Content |
|---|---|
| `id` | a name for the row |
| `messages` | a list of `{"role": ..., "content": ...}` items |

## Tasks: `env`

The server reads the sets `env/tasks` and `env/tasks-<name>`, where the name has lower-case letters, digits and hyphens. Each row is one task.

| Column | Content |
|---|---|
| `id` | the task id |
| `prompt` | the task text that the agent gets |
| `model_name` | the name that the submitted Modelica model must have |
| `spec` | the settings of the task, as JSON (see below) |
| `reference_outputs` | the reference values, as JSON: `{"time": [...], "<output>": [...], ...}` |
| `source_id` | a name for the source of the task (optional) |
| `files` | data files for the container, as JSON `{"<file name>": "<text>"}`; the agent finds them in `/work/data/` (optional) |

Other columns are allowed. The server does not read them.

The settings in `spec`:

| Key | Content |
|---|---|
| `model_name` | as above |
| `msl` | the version of the Modelica Standard Library, for example `"4.1.0"` |
| `libraries` | other libraries to load, for example `["Buildings 13.0.0"]`, or `[]` |
| `eval` | `start_time`, `stop_time`, `interval`, `tolerance` and `outputs` (the names of the required outputs) |
| `hidden_files` | library files that are removed from the container before the agent works, or `[]` |
| `hidden_classes`, `forbidden` | classes that are removed at the submission, and texts that a model must not contain (optional) |
| `limit` | the largest error of a correct output: one number, or a number for each output (optional; default 0.01) |

The reference values are compared at the fixed time steps: from `start_time` to `stop_time`, one for each `interval`. A value of `NaN` means that this time step is not compared.

The server checks each task when it starts, and it names the task and the reason when one is not right. The rules:

- A task id occurs one time only. The model name, the library names and the output names have the form of Modelica names; the times and the tolerance are numbers.
- The time from `start_time` to `stop_time` is a whole number of intervals.
- The reference values cover this time in order, and each output has one value for each time.
- A limit is a number above zero, for the task or for each output.
- A reference output is not constant, because the error is measured against its range.

## Tasks of your own

A task needs reference values. Take them from a reference model that you simulate in the task image, or from measured data.

```python
from fjallstation import image

result = image.try_model(reference_code, spec)   # one fresh container: load, check, simulate
result["status"]                                 # "simulated" when the model ran to the stop time
result["values"]                                 # the output values at the fixed time steps
```
