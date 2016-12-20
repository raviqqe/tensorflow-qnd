import sys
import unittest

import tensorflow as tf

from . import inputs
from . import test



class InputsTest(unittest.TestCase):
  def test_def_input_fn(self):
    for input_fn \
        in [inputs.def_train_input_fn(), inputs.def_eval_input_fn()]:
      # Return (tf.Tensor, tf.Tensor)

      features, labels = input_fn(lambda queue: (queue.dequeue(),) * 2)

      self.assertIsInstance(features, tf.Tensor)
      self.assertIsInstance(labels, tf.Tensor)

      # Return (dict, dict)

      features, labels = input_fn(test.user_input_fn)

      self.assertIsInstance(features, dict)
      self.assertIsInstance(labels, dict)

      self._assertAreInstances([*features.keys(), *labels.keys()], str)
      self._assertAreInstances([*features.values(), *labels.values()],
                               tf.Tensor)

  def _assertAreInstances(self, objects, klass):
    for obj in objects:
      self.assertIsInstance(obj, klass)


def append_argv():
  file_pattern = "*.md"
  sys.argv += ["--train_file", file_pattern, "--eval_file", file_pattern]


if __name__ == "__main__":
  append_argv()
  test.main()
