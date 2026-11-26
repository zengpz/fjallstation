"""nekaise: train LLM agents for the built environment.

    import nekaise

    corpus   = nekaise.Corpus()     # domain text
    instruct = nekaise.Instruct()   # chats
    env      = nekaise.Env()        # modelling tasks with a reward

The data comes from a folder that you give (`NEKAISE_STORE`). `Env()` needs the task image and the server
of this package: `nekaise image` and `nekaise server --store <folder>`.
"""

from nekaise.data import Corpus, Instruct
from nekaise.env import Env

__all__ = ["Corpus", "Instruct", "Env"]
__version__ = "0.1.0"
