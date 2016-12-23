import sys

import tensorflow as tf


def oracle_model(x, y):
    return y, 0.0, tf.no_op()


def user_input_fn(filename_queue):
    x = filename_queue.dequeue()
    return {"x": x}, {"y": x}


def append_argv(*args):
    command = "THIS_SHOULD_NEVER_MATCH"

    if sys.argv[0] != command:
        sys.argv = [command]

    sys.argv += [*args]
