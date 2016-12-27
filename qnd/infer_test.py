import types

from .estimator_test import append_argv
from . import infer
from . import test


def test_def_infer():
    append_argv()
    test.append_argv("--output_dir", "output")
    assert isinstance(infer.def_infer(), types.FunctionType)
