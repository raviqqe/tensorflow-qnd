import types
import unittest

import tensorflow as tf

from . import test
from .experiment import *
from . import estimator_test
from . import inputs_test



class ExperimentTest(unittest.TestCase):
  def test_def_experiment(self):
    def_experiment_fn = def_def_experiment_fn()
    self._assertIsFunction(def_experiment_fn)

    experiment_fn = def_experiment_fn(test.oracle_model, test.user_input_fn)
    self._assertIsFunction(experiment_fn)

    self.assertIsInstance(experiment_fn("output"), tf.contrib.learn.Experiment)

  def _assertIsFunction(self, obj):
    self.assertIsInstance(obj, types.FunctionType)


def append_argv():
  for sub_test in [estimator_test, inputs_test]:
    sub_test.append_argv()


if __name__ == "__main__":
  append_argv()
  test.main()
