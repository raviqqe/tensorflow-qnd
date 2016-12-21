import unittest

import tensorflow as tf

from .test import *



class TestTest(unittest.TestCase):
  def test_oracle_model(self):
    oracle_model(tf.zeros([100]), tf.zeros([100]))

  def test_user_input_fn(self):
    user_input_fn(tf.FIFOQueue(64, [tf.string]))



if __name__ == "__main__":
  unittest.main()
