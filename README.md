# tensorflow-qnd

[![PyPI version](https://badge.fury.io/py/tensorflow-qnd.svg)](https://badge.fury.io/py/tensorflow-qnd)
[![Python versions](https://img.shields.io/pypi/pyversions/tensorflow-qnd.svg)]()
[![Build Status](https://travis-ci.org/raviqqe/tensorflow-qnd.svg?branch=master)](https://travis-ci.org/raviqqe/tensorflow-qnd)
[![License](https://img.shields.io/badge/license-unlicense-lightgray.svg)](https://unlicense.org)

Quick and Distributed TensorFlow command framework

tensorflow-qnd is a TensorFlow framework to create commands to train and
evaluate models on multiple computers.
While made to be used on multiple computers in a cluster, this library is also
useful to exploit multiple GPUs on a single machine.


## Installation

Python 3.5+ and TensorFlow 0.12+ are required.

```
pip3 install --user --upgrade tensorflow-qnd
```


## Usage

See [documentation](https://raviqqe.github.io/tensorflow-qnd/qnd).


## Examples

```python
import logging

import qnd
import tensorflow as tf


logging.getLogger().setLevel(logging.INFO)


def read_file(filename_queue):
    _, serialized = tf.TFRecordReader().read(filename_queue)

    scalar_feature = lambda dtype: tf.FixedLenFeature([], dtype)

    features = tf.parse_single_example(serialized, {
        "image_raw": scalar_feature(tf.string),
        "label": scalar_feature(tf.int64),
    })

    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image.set_shape([28**2])

    return tf.to_float(image) / 255 - 0.5, features["label"]


def minimize(loss):
    return tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        0.01,
        "Adam")


def model(image, number):
    h = tf.contrib.layers.fully_connected(image, 64)
    h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(h, number))
    predictions = tf.argmax(h, axis=1)

    return predictions, loss, minimize(loss), {
        "accuracy": tf.reduce_mean(tf.to_float(tf.equal(predictions, number)))
    }


run = qnd.def_run()


def main():
    run(model, read_file)


if __name__ == "__main__":
    main()
```

See also [examples](examples) directory.


## License

[The Unlicense](https://unlicense.org)
