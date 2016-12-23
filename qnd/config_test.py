import tensorflow as tf

from . import test
from . import config


def test_def_config():
    append_argv()
    assert isinstance(config.def_config()(), tf.contrib.learn.ClusterConfig)


def append_argv():
    test.append_argv("--master_host", "localhost:4242",
                     "--ps_hosts", "localhost:5151",
                     "--task_type", "master")
