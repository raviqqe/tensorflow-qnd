import functools

import tensorflow as tf



def func_scope(func):
  @functools.wraps(func)
  def wrapped_func(*args, **kwargs):
    with tf.variable_scope(func.__name__):
      return func(*args, **kwargs)
  return wrapped_func
