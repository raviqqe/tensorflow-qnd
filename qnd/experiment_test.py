import unittest

import tensorflow as tf
from gargparse import ARGS

from . import test
from .experiment import *
from . import estimator_test
from . import inputs_test



class ExperimentTest(unittest.TestCase):
  def test_def_experiment(self):
    experiment = def_experiment()
    self.assertIsInstance(experiment(test.oracle_model, test.user_input_fn),
                          tf.contrib.learn.Experiment)


if __name__ == "__main__":
  for sub_test in [estimator_test, inputs_test]:
    sub_test.append_argv()
  test.main()
