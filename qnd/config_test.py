import sys
import unittest

import tensorflow as tf

from . import test
from .config import *



class ConfigTest(unittest.TestCase):
  def test_def_config(self):
    config = def_config()
    self.assertTrue(isinstance(config(), tf.contrib.learn.ClusterConfig))


def append_argv():
  sys.argv += [
      "--ps_hosts", "localhost:4242",
      "--worker_hosts", "localhost:5353",
      "--task_type", "ps",
      "--task_index", "0"]


if __name__ == "__main__":
  append_argv()
  test.main()
