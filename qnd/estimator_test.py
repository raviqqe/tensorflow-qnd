import unittest

import tensorflow as tf

from . import test
from .config_test import append_argv
from .estimator import *



class EstimatorTest(unittest.TestCase):
  def test_def_estimator(self):
    self.assertTrue(isinstance(def_estimator()(test.oracle_model),
                    tf.contrib.learn.Estimator))


if __name__ == "__main__":
  append_argv()
  test.main()
