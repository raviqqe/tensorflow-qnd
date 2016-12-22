import functools
import inspect

import tensorflow as tf


def func_scope(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        with tf.variable_scope(func.__name__):
            return func(*args, **kwargs)

    # inspect.getargspec() (used in TensorFlow) cannot be deceived by
    # functools.wraps() somehow. So we need to assign a signature of an original
    # function to a wrapper. This can be a bug of Python.
    wrapped_func.__signature__ = inspect.signature(func)
    return wrapped_func


def are_instances(objects, klass):
    return all(isinstance(obj, klass) for obj in objects)
