import unittest

from .util import *



class TestUtil(unittest.TestCase):
  def test_func_scope(self):
    @func_scope
    def foo():
      pass


if __name__ == "__main__":
  unittest.main()
