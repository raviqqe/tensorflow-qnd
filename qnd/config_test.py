import tensorflow as tf

from . import test
from . import config


TEST_ARGS = ["--master_host", "localhost:4242",
             "--ps_hosts", "localhost:5151",
             "--task_type", "master"]


def test_def_config():
    test.initialize_argv(*TEST_ARGS)
    assert isinstance(config.def_config()(), tf.contrib.learn.ClusterConfig)
