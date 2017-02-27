import tensorflow as tf

from . import test
from . import config


def test_def_config():
    test.append_argv()
    assert isinstance(config.def_config()(), tf.contrib.learn.RunConfig)
