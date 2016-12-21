import sys
import unittest

import tensorflow as tf



def main():
  unittest.main(argv=sys.argv[:1])


def oracle_model(x, y):
  return y, 0.0, tf.no_op()


def user_input_fn(filename_queue):
  x = filename_queue.dequeue()
  return {"x": x}, {"y": x}
