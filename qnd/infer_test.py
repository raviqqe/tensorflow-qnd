import types

from . import infer
from . import test


def test_def_infer():
    test.append_argv("--output_dir", "output")
    assert isinstance(infer.def_infer(), types.FunctionType)
