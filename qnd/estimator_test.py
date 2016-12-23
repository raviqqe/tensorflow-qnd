import tensorflow as tf

from . import test
from .config_test import TEST_ARGS
from .estimator import *


def test_def_estimator():
    test.initialize_argv(*TEST_ARGS)
    assert isinstance(def_estimator()(test.oracle_model, "output"),
                      tf.contrib.learn.Estimator)
    assert isinstance(
        def_estimator()(
            tf.contrib.learn.estimators.model_fn.ModelFnOps(test.oracle_model),
            "output"),
        tf.contrib.learn.Estimator)
