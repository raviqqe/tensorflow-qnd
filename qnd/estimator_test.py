import unittest

import tensorflow as tf

from . import test
from . import config_test
from .estimator import *



class EstimatorTest(unittest.TestCase):
  def test_def_estimator(self):
    self.assertTrue(isinstance(def_estimator()(oracle_model),
                    tf.contrib.learn.Estimator))


def oracle_model(x, y):
  return y, 0.0, tf.no_op()


if __name__ == "__main__":
  config_test.append_argv()
  test.main()
