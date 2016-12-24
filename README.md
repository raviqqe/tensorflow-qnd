# tensorflow-qnd

[![PyPI version](https://badge.fury.io/py/tensorflow-qnd.svg)](https://badge.fury.io/py/tensorflow-qnd)
[![Python versions](https://img.shields.io/pypi/pyversions/tensorflow-qnd.svg)]()
[![Build Status](https://travis-ci.org/raviqqe/tensorflow-qnd.svg?branch=master)](https://travis-ci.org/raviqqe/tensorflow-qnd)
[![License](https://img.shields.io/badge/license-unlicense-lightgray.svg)](https://unlicense.org)

Quick and Distributed TensorFlow command framework

tensorflow-qnd is a TensorFlow framework to create commands to experiment with
models on multiple computers.
While made to be used on multiple computers in a cluster, this library is also
useful to exploit multiple GPUs on a single machine.


## Installation

Python 3.5+ and TensorFlow 0.12+ are required.

```
$ pip3 install --user --upgrade tensorflow-qnd
```


## Usage

```
def_run(batch_inputs=True, prepare_filename_queues=True)
    Define `run()` function.

    See also `help(def_run())`.

    Args:
        batch_inputs: If `True`, create batches from Tensors
            returned from `train_input_fn()` and `train_input_fn()` and feed
            them to a model.
        prepare_filename_queues: If `True`, create filename queues for train
            and eval data based on file paths specified by command line
            arguments.

    Returns:
        `run()` function.


run(model_fn, train_input_fn, eval_input_fn=None)
    Run `tf.contrib.learn.python.learn.learn_runner.run()`.

    Args:
        model_fn: A function to construct a model.
            Types of its arguments must be one of the following:
                `Tensor, ...`,
                `Tensor, ..., mode=ModeKeys`.
            Types of its return values must be one of the following:
                `Tensor, Tensor, Operation, eval_metric_ops=dict<str, Tensor>`
                (predictions, loss, train_op, and eval_metric_ops (if any)),
                `ModelFnOps`.
        train_input_fn, eval_input_fn: Functions to serve input Tensors
            fed into the model. If `eval_input_fn` is `None`,
            `train_input_fn` will be used instead.
            Types of its arguments must be one of the following:
                `QueueBase` (a filename queue),
                `None` (No argument).
            Types of its return values must be one of the following:
                `Tensor, Tensor` (x and y),
                `dict<str, Tensor>, dict<str, Tensor>`
                (features and labels).
            The keys of `dict` arguments must match with names of
            `model_fn` arguments.

    Returns:
        Return value of `tf.contrib.learn.python.learn.learn_runner.run()`.


add_flag(name, *args, **kwargs)
    Add a flag.

    Added flags can be accessed by `FLAGS` module variable.
    (e.g. `FLAGS.my_flag_name`)

    Args:
        name: Flag name. Real flag name will be `"--{}".format(name)`.
        *args, **kwargs: The rest arguments are the same as
            `argparse.add_argument()`.


add_required_flag(name, *args, **kwargs)
    Add a required flag.

    Its interface is the same as `add_flag()` but `required=True` is set by
    default.
```

For more information, see [documentation](https://raviqqe.github.io/tensorflow-qnd/qnd).


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
