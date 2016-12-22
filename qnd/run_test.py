import types

from . import test
from .experiment_test import TEST_ARGS
from . import run


def test_def_run():
    test.initialize_argv(TEST_ARGS)
    assert isinstance(run.def_run(), types.FunctionType)
