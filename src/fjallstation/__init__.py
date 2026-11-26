"""fjallstation: a data and RL environment API for LLM training in the built environment.

    import fjallstation

    corpus   = fjallstation.Corpus()     # domain text
    instruct = fjallstation.Instruct()   # chats
    env      = fjallstation.Env()        # modelling tasks with a reward

The data comes from a folder that you give (`FJALLSTATION_STORE`). `Env()` needs the task image and the server
of this package: `fjallstation image` and `fjallstation server --store <folder>`.
"""

from fjallstation.data import Corpus, Instruct
from fjallstation.env import Env

__all__ = ["Corpus", "Instruct", "Env"]
__version__ = "0.1.1"
