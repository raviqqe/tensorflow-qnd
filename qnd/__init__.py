"""Quick and Distributed TensorFlow command framework"""

from .flag import *
from .run import def_run

__all__ = ["FLAGS", "add_flag", "add_required_flag", "FlagAdder", "def_run"]
__version__ = "0.0.5"
