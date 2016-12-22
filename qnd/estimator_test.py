import tensorflow as tf

from . import test
from .config_test import TEST_ARGS
from .estimator import *


def test_def_estimator():
    test.initialize_argv(*TEST_ARGS)
    assert isinstance(def_estimator()(test.oracle_model, "output"),
                      tf.contrib.learn.Estimator)
