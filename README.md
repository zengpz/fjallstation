# Fjällstation

*"Fjällstation is the mountain station at the foot of Kebnekaise, where you prepare before the climb."*

A data and RL environment API for LLM training in the built environment.

## What you get

| Call | What the model gets | For |
|---|---|---|
| `fjallstation.Corpus()` | text to read: domain knowledge | training on raw text |
| `fjallstation.Instruct()` | chats to follow: questions with answers | supervised fine-tuning (SFT) |
| `fjallstation.Env()` | tasks to do: modelling work with a reward | reinforcement learning (RL) |

`Env()` runs on two more parts of the package: a task image (OpenModelica with its libraries) and a server (it gives the tasks and the rewards).

You bring the data: one folder in a [simple format](https://github.com/zengpz/fjallstation/blob/main/docs/data.md). The same parts run on one workstation or on several machines.

## Quick start

You need Python 3.10 or later and Docker. Tested on Linux.

```bash
pip install fjallstation
fjallstation image                    # build the task image; the first build takes some minutes
fjallstation example ./store          # the example store: five texts, five chats, five tasks
fjallstation server --store ./store   # the server; keep it running in this terminal
```

Then your training code uses it. A mock-up: the model, the agent and the training steps are yours.

```python
import fjallstation

llm = load_your_llm()

# 1. Training on raw domain text
for doc in fjallstation.Corpus():
    llm.train_on_text(doc["text"])

# 2. SFT on chats
for row in fjallstation.Instruct():
    llm.train_on_chat(row["messages"])

# 3. RL on tasks with a reward
env = fjallstation.Env()
for episode in range(1000):
    task = env.reset()                             # a task and a fresh container
    agent = Agent(llm, task.prompt)                # your agent: the LLM with a shell
    while agent.works():
        output = env.exec(agent.next_command()).output
        agent.read(output)
    result = env.step(agent.final_model())         # the container tries the model; the server scores it
    llm.learn(agent.steps, result.reward)          # 1.0 for a correct model; 0 to 0.3 for how far another one got
```

A [first episode that you can run](https://github.com/zengpz/fjallstation/blob/main/docs/protocol.md) is in the documents.

## Documents

- [The data folder](https://github.com/zengpz/fjallstation/blob/main/docs/data.md): the format, the examples, and how to make tasks of your own.
- [One episode, step by step](https://github.com/zengpz/fjallstation/blob/main/docs/protocol.md): a first episode with Python and with a shell, who runs as whom, and use on several machines.
- [The reward](https://github.com/zengpz/fjallstation/blob/main/docs/scoring.md): the two parts of the reward and the rule behind them.

## Licence

Apache 2.0, see [LICENSE](https://github.com/zengpz/fjallstation/blob/main/LICENSE). The task image installs OpenModelica and Modelica libraries; they have their own licences. Two example tasks use open datasets under CC BY 4.0; their credit is in the task text and in the data document.
