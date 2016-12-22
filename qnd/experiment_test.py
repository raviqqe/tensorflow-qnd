import types

import tensorflow as tf

from . import test
from . import estimator_test
from . import experiment
from . import inputs_test


TEST_ARGS = [*estimator_test.TEST_ARGS, *inputs_test.TEST_ARGS]


def test_def_experiment():
    test.initialize_argv(*TEST_ARGS)

    def_experiment_fn = experiment.def_def_experiment_fn()
    _assert_is_function(def_experiment_fn)

    experiment_fn = def_experiment_fn(test.oracle_model, test.user_input_fn)
    _assert_is_function(experiment_fn)

    assert isinstance(experiment_fn("output"), tf.contrib.learn.Experiment)


def _assert_is_function(obj):
    assert isinstance(obj, types.FunctionType)
