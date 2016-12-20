# qnd.tf

[![Build Status](https://travis-ci.org/raviqqe/qnd.tf.svg?branch=master)](https://travis-ci.org/raviqqe/qnd.tf)
[![License](https://img.shields.io/badge/license-unlicense-lightgray.svg)](https://unlicense.org)

Quick and Distributed TensorFlow command framework

qnd.tf is a framework to create commands to experiment models
on distributed systems.


## Installation

```
$ pip install --user --upgrade tensorflow-qnd
```


## Usage

WIP


## Examples

WIP


## Design

All you need to do is to define model and input functions.

- Model function
  - Python function : features and labels... -> predictions, loss, train op, eval metrics (if any)
- Input function
  - tf.ReaderBase
  - Python function : filename -> keys, features and labels...
    - The first dimension of each features and labels should be number of samples inside.


## License

[The Unlicense](https://unlicense.org)
