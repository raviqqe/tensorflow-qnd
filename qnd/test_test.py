import tensorflow as tf

from .test import *


def test_oracle_model():
    oracle_model(tf.zeros([100]), tf.zeros([100]))


def test_user_input_fn():
    user_input_fn(tf.FIFOQueue(64, [tf.string]))
