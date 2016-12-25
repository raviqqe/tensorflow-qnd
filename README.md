# tensorflow-qnd

[![PyPI version](https://badge.fury.io/py/tensorflow-qnd.svg)](https://badge.fury.io/py/tensorflow-qnd)
[![Python versions](https://img.shields.io/pypi/pyversions/tensorflow-qnd.svg)]()
[![Build Status](https://travis-ci.org/raviqqe/tensorflow-qnd.svg?branch=master)](https://travis-ci.org/raviqqe/tensorflow-qnd)
[![License](https://img.shields.io/badge/license-unlicense-lightgray.svg)](https://unlicense.org)

Quick and Distributed TensorFlow command framework

tensorflow-qnd is a TensorFlow framework to create commands to train and
evaluate models on multiple computers.
The framework is built on top of
`tensorflow.contrib.learn.python.learn.learn_runner` and relevant modules.
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
        0.001,
        "Adam")


def model(image, number):
    h = tf.contrib.layers.fully_connected(image, 64)
    h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(h, number))
    predictions = tf.argmax(h, axis=1)

    return predictions, loss, minimize(loss), {
        "accuracy": tf.contrib.metrics.streaming_accuracy(predictions,
                                                          number)[1],
    }


run = qnd.def_run()


def main():
    run(model, read_file)


if __name__ == "__main__":
    main()
```

With the code above, you can create a command with the following interface.

```
usage: mnist_simple.py [-h] [--output_dir OUTPUT_DIR]
                       [--train_steps TRAIN_STEPS] [--eval_steps EVAL_STEPS]
                       [--min_eval_frequency MIN_EVAL_FREQUENCY] --master_host
                       MASTER_HOST --ps_hosts PS_HOSTS
                       [--worker_hosts WORKER_HOSTS] --task_type TASK_TYPE
                       [--task_index TASK_INDEX] [--num_cores NUM_CORES]
                       [--log_device_placement]
                       [--save_summary_steps SAVE_SUMMARY_STEPS]
                       [--save_checkpoints_steps SAVE_CHECKPOINTS_STEPS]
                       [--batch_size BATCH_SIZE]
                       [--batch_queue_capacity BATCH_QUEUE_CAPACITY]
                       --train_file TRAIN_FILE
                       [--filename_queue_capacity FILENAME_QUEUE_CAPACITY]
                       --eval_file EVAL_FILE

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Directory where checkpoint and event files are stored
                        (default: output)
  --train_steps TRAIN_STEPS
                        Maximum number of train steps (default: None)
  --eval_steps EVAL_STEPS
                        Maximum number of eval steps (default: None)
  --min_eval_frequency MIN_EVAL_FREQUENCY
                        Minimum evaluation frequency in number of model
                        savings (default: 1)
  --master_host MASTER_HOST
                        $hostname:$port pair of a master host (default: None)
  --ps_hosts PS_HOSTS   Comma-separated list of $hostname:$port pairs of ps
                        hosts (default: [])
  --worker_hosts WORKER_HOSTS
                        Comma-separated list of $hostname:$port pairs of
                        worker hosts (default: [])
  --task_type TASK_TYPE
                        Must be in ['master', 'ps', 'worker'] (aka job)
                        (default: None)
  --task_index TASK_INDEX
                        Task index within a job (default: 0)
  --num_cores NUM_CORES
                        Number of CPU cores used. 0 means use of a default
                        value. (default: 0)
  --log_device_placement
                        If specified, log device placement information
                        (default: False)
  --save_summary_steps SAVE_SUMMARY_STEPS
                        Number of steps every time of which summary is saved
                        (default: 100)
  --save_checkpoints_steps SAVE_CHECKPOINTS_STEPS
                        Number of steps every time of which a model is saved
                        (default: None)
  --batch_size BATCH_SIZE
                        Mini-batch size (default: 64)
  --batch_queue_capacity BATCH_QUEUE_CAPACITY
                        Batch queue capacity (default: 1024)
  --train_file TRAIN_FILE
                        File path of train data file(s). A glob is available.
                        (e.g. train/*.tfrecords) (default: None)
  --filename_queue_capacity FILENAME_QUEUE_CAPACITY
                        Capacity of filename queues of train and eval data
                        (default: 32)
  --eval_file EVAL_FILE
                        File path of eval data file(s). A glob is available.
                        (e.g. eval/*.tfrecords) (default: None)
```

See also [examples](examples) directory.


## License

[The Unlicense](https://unlicense.org)
