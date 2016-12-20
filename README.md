# qnd.tf

[![Build Status](https://travis-ci.org/raviqqe/qnd.tf.svg?branch=master)](https://travis-ci.org/raviqqe/qnd.tf)
[![License](https://img.shields.io/badge/license-unlicense-lightgray.svg)](https://unlicense.org)

Quick and Distributed TensorFlow command framework

qnd.tf is a framework to create commands to experiment models
on distributed systems.


## Installation

Python 3.5+ is required.

```
$ pip install --user --upgrade tensorflow-qnd
```


## Usage

WIP


## Examples

WIP


## Design

All users need to do is to define model and file decoder functions.

- Model function : features and labels... -> predictions, loss, train op, eval metrics (if any)
- File decoder function : filename queue -> features, labels
  - Features and labels can be a `dict` of `str` to `tf.Tensor` or a single `tf.Tensor`.
  - Features and labels should be batched.
  - Its input is a filename queue.
    - To let users use [Readers](https://www.tensorflow.org/api_docs/python/io_ops/readers#FixedLengthRecordReader).


## License

[The Unlicense](https://unlicense.org)
