import tensorflow as tf

from . import test
from .estimator import *


def test_def_estimator():
    test.append_argv()
    assert isinstance(def_estimator()(test.oracle_model, "output"),
                      tf.contrib.learn.Estimator)
    assert isinstance(
        def_estimator()(
            lambda x, y: tf.contrib.learn.ModelFnOps(
                "train", *test.oracle_model(x, y)),
            "output"),
        tf.contrib.learn.Estimator)
