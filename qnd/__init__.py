"""Quick and Distributed TensorFlow command framework"""

from .flag import *
from .train_and_evaluate import def_train_and_evaluate

__all__ = ["FLAGS", "add_flag", "add_required_flag", "FlagAdder",
           "def_train_and_evaluate"]
__version__ = "0.0.8"
