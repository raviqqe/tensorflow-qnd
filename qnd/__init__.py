"""Quick and Dirty TensorFlow command framework"""

from .flag import *
from .infer import def_infer
from .train_and_evaluate import def_train_and_evaluate
from .evaluate import def_evaluate
from .serve import def_serve

__all__ = ["FLAGS", "add_flag", "add_required_flag", "FlagAdder",
           "def_train_and_evaluate", "def_evaluate", "def_infer", "def_serve"]
__version__ = "0.1.6"
