# The data folder

All data is in one folder, the store. `FJALLSTATION_STORE` names it for `Corpus()` and `Instruct()`; `fjallstation server --store` names it for the server. The default of both is `./store`.

```
store/
  corpus/<set>/part-*.parquet       text
  instruct/<set>/part-*.parquet     chats
  env/tasks-<name>/part-*.parquet   tasks of Env
```

A set is a folder with one or more Parquet part files. You can have many sets and any of the three parts.

## The example store

`fjallstation example ./store` copies a small store of this form into the folder: five texts, five chats and five tasks. The files are a part of the package.

| Part | Examples | Source |
|---|---|---|
| `corpus/example` | five short lessons: heat loss of district-heating pipes, the COP of a heat pump, equations in Modelica, a CO₂ balance of an office, and one lesson in Swedish about a radiator | written by a language model and reviewed |
| `instruct/example` | five chats with one question and one answer: a heat pump, units in Modelica, a district-heating substation, the response of a room, and one chat in Swedish | written by a language model and reviewed |
| `env/tasks-example` | five tasks, see below | |

| Task id | What the agent builds | Kind |
|---|---|---|
| `example-heated-room` | a room as one heat capacity with a heater | a small task with every number in the text |
| `example-heating-curve` | supply and return temperature setpoints with blocks of the Buildings library | a task from a library model; the library files with the answer are hidden |
| `example-pid-controller` | a comparison of PID controllers of two libraries | a task from a library model |
| `example-return-pipe` | the outlet temperature of a district-heating pipe | a task from measured data, with data files and a limit of its own |
| `example-heat-pump` | the electric energy of an air-to-water heat pump | a task from measured data |

The two tasks from measured data use open datasets under CC BY 4.0. The task text ends with the credit and says what was changed.

- `example-return-pipe`: IMOS-SWIL District Heating Dataset, version 1.0.0, by K. Faghih Niresi, C. Møller Jensen, C. S. Kallesøe, R. Wisniewski and O. Fink (2026), https://doi.org/10.5281/zenodo.23127439.
- `example-heat-pump`: High-Resolution Datasets of Thermal Dynamics, Heat Pump Operation, and Grid Interaction in Three Experimental Buildings, by J. Wachter and others, Karlsruhe Institute of Technology (2026), https://doi.org/10.5281/zenodo.21157950.

## Text: `corpus`

`Corpus()` gives the rows one after the other, as a streaming `datasets` object of Hugging Face.

| Column | Content |
|---|---|
| `id` | a name for the row |
| `text` | the text |
| `source` | where the text comes from, in one word |

## Chats: `instruct`

`Instruct()` gives all rows, as a `datasets` object of Hugging Face.

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
- Task files map plain filenames to text. Names must be nonempty, must not start with a dot, and must not contain `/` or a NUL character. The task creator is responsible for the contents; the server does not parse or validate the datasets inside these files.

## Tasks of your own

A task needs reference values. Take them from a reference model that you simulate in the task image, or from measured data. Then write the rows as a Parquet file into a folder `env/tasks-<name>`.

```python
from fjallstation import image

result = image.try_model(reference_code, spec)   # one fresh container: load, check, simulate
result["status"]                                 # "simulated" when the model ran to the stop time
result["values"]                                 # the output values at the fixed time steps
```
