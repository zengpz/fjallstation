# The reward

The reward has two parts. `Env.step()` returns both.

## The OpenModelica part

It comes from the container and says how far the model got.

| The model | Status | Value |
|---|---|---|
| is empty, does not load, or uses a forbidden text | `no model`, `did not load`, `uses the source example` | 0 |
| loads, and the check fails | `check failed` | 0.1 |
| checks, and the simulation does not reach the stop time | `simulation failed`, `simulation stopped early`, `time limit` | 0.2 |
| simulates to the stop time | `simulated` | 0.3 |

The time limit for one submission is 30 minutes. `time limit` has the value 0.2 also when the limit ends the load or the check.

## The scorer part

It comes from the server. The server compares each required output with the reference values at the fixed time steps.

- The error of an output is its NRMSE: the root mean square error divided by the range of the reference values.
- The model is correct when every output is at or below its limit. The limit is 0.01, or the value that the task names in `spec["limit"]`.
- A time step with no reference value is not compared.
- A reference output must not be constant, because its range is the measure. The server refuses such a task when it starts.

| Status | Meaning | Scorer part |
|---|---|---|
| `correct` | every output is inside its limit | 1.0 |
| `outputs differ` | an output is outside its limit | 0 |
| `output missing` | the model has no variable with a required name | 0 |
| `result not valid` | the result file does not have the right form | 0 |

## The reward

`result.reward` is 1.0 for a correct model. For every other model it is the OpenModelica part.

```python
result = env.step(model_code)
result.reward     # 1.0, or 0, 0.1, 0.2, 0.3
result.omc        # the OpenModelica part
result.scorer     # the scorer part
result.feedback   # the status
```

## What the reward is for

The container of the user simulates the model, and the server does not simulate it again. So the reward is a training signal. It is not a trusted score: a person who has the data folder has the reference values too.

The rule is in one file, `src/nekaise/score.py`.
