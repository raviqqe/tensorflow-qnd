import types

from .experiment_test import append_argv
from . import train_and_evaluate


def test_def_train_and_evaluate():
    append_argv()
    assert isinstance(train_and_evaluate.def_train_and_evaluate(),
                      types.FunctionType)
