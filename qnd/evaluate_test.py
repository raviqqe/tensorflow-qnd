import types

from . import evaluate
from . import test


def test_def_evaluate():
    test.append_argv("--output_dir", "output")
    assert isinstance(evaluate.def_evaluate(), types.FunctionType)
