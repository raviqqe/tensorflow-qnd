import types

import tensorflow as tf

from . import test
from . import experiment
from . import inputs_test


def test_def_experiment():
    append_argv()

    def_experiment_fn = experiment.def_def_experiment_fn()
    _assert_is_function(def_experiment_fn)

    experiment_fn = def_experiment_fn(test.oracle_model, test.user_input_fn)
    _assert_is_function(experiment_fn)

    assert isinstance(experiment_fn("output"), tf.contrib.learn.Experiment)


def _assert_is_function(obj):
    assert isinstance(obj, types.FunctionType)


def append_argv():
    inputs_test.append_argv()
