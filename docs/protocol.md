# One episode, step by step

`fjallstation.Env()` is a small client. It does four steps that any program can do, in any language. The container needs no network.

| Step | Who | What |
|---|---|---|
| 1 | your program | `POST /v1/tasks/sample` to the server. The answer is the task file. |
| 2 | your program | Start a container of the task image and give it the task file: `fjallstation-task start`. |
| 3 | the agent | Works in the container, in `/work`, as a normal user. |
| 4 | your program | `fjallstation-task submit` in the container tries the model and prints the result file. Send it to the address in it. The answer holds the reward. |

## A first episode with Python

With the example store and the server of the quick start:

```python
import fjallstation                       # reads ./store, or the folder in FJALLSTATION_STORE

for doc in fjallstation.Corpus():
    print(doc["text"])
for row in fjallstation.Instruct():
    print(row["messages"])

env = fjallstation.Env()
task = env.reset("example-heated-room")  # a task and a fresh container; env.reset() gives a random task
print(task.prompt)
print(env.exec("omc --version").output)  # your agent works in the container

model = """
model HeatedRoom
  Real T(start = 10, fixed = true);
  output Real T_room;
equation
  2e6*der(T) = 1500 - 100*T;
  T_room = T;
end HeatedRoom;
"""
result = env.step(model)                 # the container tries the model; the server gives the reward
print(result.reward, result.feedback)    # 1.0 correct
```

## With a shell

```bash
curl -s -X POST http://127.0.0.1:8040/v1/tasks/sample -d '{}' > task.json
docker run -d --name episode --network none --pids-limit 512 --user 1000:1000 fjallstation-env:2
docker exec -i -u 0 episode fjallstation-task start < task.json

# ... the agent works: docker exec -w /work episode bash -c "<command>" ...

docker exec -i -u 0 episode fjallstation-task submit < model.mo > result.json
docker rm -f episode
curl -s -X POST "$(python3 -c 'import json; print(json.load(open("result.json"))["address"])')" -d @result.json
```

## Who runs as whom

| Who | User in the container | Rights |
|---|---|---|
| the agent | 1000 | its own files in `/work` and `/tmp`; no network |
| `fjallstation-task start` and `submit` | root | the library folder and the kept task file; Python runs in isolated mode, so it reads no start-up file of the agent |
| the submitted model (compilation and simulation) | 65534 | a folder of its own; a model that runs a shell command has no rights of root or of the agent |

Stop the processes of the agent before the submission. The container is the outer boundary: an episode mounts no folder of the host. `fjallstation.Env()` gives the container limits for CPU, memory and the number of processes. It gives no limit for the disk.

`env.exec(command)` stops a command at its time limit (exit code 124) and returns the first megabyte of its output. The command keeps its own exit code. A process that stays in the background must write to a file. These limits are a guard against accidents, not against an agent that works against them.

## The server

| Call | Body | Answer |
|---|---|---|
| `POST /v1/tasks/sample` | `{}` for a random task, or `{"task": "<task id>"}` | the task file |
| `POST /v1/results` | the result file | `{"reward", "omc_reward", "status", "task", "image"}` |
| `GET /v1/health` | | `{"status": "ok", "tasks": <count>, "image"}` |

The server retains up to 200,000 ticket records in memory, counting both active and completed episodes. This is a record limit, not a measure of supported training throughput. When space is needed, it removes the oldest completed record. It keeps open tickets and tickets currently being scored. If all slots are active, a new task request receives HTTP 503 with a clear busy message; existing workers can still submit results. Retry the task request after a slot becomes available.

Removing a completed ticket removes its cached reply, not the saved reward log. A later retry for that removed ticket receives HTTP 404. Active tickets have no automatic expiry, so abandoned episodes can occupy capacity. All tickets are lost on a server restart.

## The two files

The task file: `ticket`, `address`, `task`, `prompt`, `model_name`, `image`, `hidden_files`, `run` (the settings to try a model) and, for a task with data files, `files`.

The result file: `ticket`, `address`, `task`, `image`, `status`, `omc_reward`, `values` (the required outputs at the fixed time steps) and `seconds`.

- The ticket names one task instance. The server takes one result for each ticket.
- The address is where the result goes. The server writes its own address into the task file, so nothing is configured.
- The server refuses a result from another image version, because other versions can give other values.
- The reference values and the error of each output stay on the server.

## When a result cannot be sent

`env.step()` raises the error, and the `Env` keeps the result file. `env.send()` sends the file again; no simulation runs again, and the server gives the same answer for the same result while it has the ticket. The next `env.reset()` drops a result that was not sent: that episode is lost, and the worker goes on. Use one `Env` for each worker.

## On several machines

- Start the server with `fjallstation server --host 0.0.0.0`, and give the clients its address: `fjallstation.Env(url="http://<server>:8040")` or the variable `FJALLSTATION_SERVER`.
- Each machine that runs containers needs the task image (`fjallstation image`).
- `fjallstation.Env()` sends the result to the server address that you gave it. The address in the task file is for a program with no configuration; behind a proxy or with HTTPS it is not right.
- The server has no accounts, no encryption and no request rate limits. Use it on a machine or a network that you trust. It keeps the open tickets in memory, so a restart ends the open episodes.
- Treat a task set from another person as a program: its settings go into the script that the container runs.
