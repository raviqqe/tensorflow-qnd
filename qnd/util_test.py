from .util import *


def test_func_scope():
    @func_scope
    def foo():
        pass
