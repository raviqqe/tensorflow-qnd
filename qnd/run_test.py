import types
import unittest

from . import test
from .experiment_test import append_argv
from .run import *



class RunTest(unittest.TestCase):
  def test_def_run(self):
    self.assertIsInstance(def_run(), types.FunctionType)


if __name__ == "__main__":
  append_argv()
  test.main()
