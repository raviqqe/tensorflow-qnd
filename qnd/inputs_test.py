import tensorflow as tf

from . import inputs
from . import test


_FILE_PATTERN = "*.md"


def test_def_input_fn():
    append_argv()

    for def_input_fn in [inputs.def_def_train_input_fn(),
                         inputs.def_def_eval_input_fn()]:
        # Return (tf.Tensor, tf.Tensor)

        features, labels = def_input_fn(lambda queue: (queue.dequeue(),) * 2)()

        assert isinstance(features, tf.Tensor)
        assert isinstance(labels, tf.Tensor)

        # Return (dict, dict)

        features, labels = def_input_fn(test.user_input_fn)()

        assert isinstance(features, dict)
        assert isinstance(labels, dict)

        _assert_are_instances([*features.keys(), *labels.keys()], str)
        _assert_are_instances(
            [*features.values(), *labels.values()], tf.Tensor)


def _assert_are_instances(objects, klass):
    for obj in objects:
        assert isinstance(obj, klass)


def append_argv():
    test.append_argv("--train_file", _FILE_PATTERN,
                     "--eval_file", _FILE_PATTERN)
