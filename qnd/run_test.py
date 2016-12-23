import types

from .experiment_test import append_argv
from . import run


def test_def_run():
    append_argv()
    assert isinstance(run.def_run(), types.FunctionType)
