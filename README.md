# tensorflow-qnd

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

WIP


## Examples

See [examples](examples) directory.


## Design

All users need to do is to define model and input functions.

- Model function : features and labels... -> predictions, loss, train op, eval metrics (if any)
- Input function : filename queue -> features, labels
  - Features and labels can be a `dict` of `str` to `tf.Tensor` or a single `tf.Tensor`.
  - Features and labels should be batched.
  - Its input is a filename queue so that users can use [Readers](https://www.tensorflow.org/api_docs/python/io_ops/readers#FixedLengthRecordReader).


## License

[The Unlicense](https://unlicense.org)
