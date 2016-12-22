import sys

import tensorflow as tf


def oracle_model(x, y):
    return y, 0.0, tf.no_op()


def user_input_fn(filename_queue):
    x = filename_queue.dequeue()
    return {"x": x}, {"y": x}


def initialize_argv(*args):
    sys.argv = [sys.argv[0], *args]
