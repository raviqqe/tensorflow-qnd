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

```
def_run(batch_inputs=True, prepare_filename_queues=True)
    Define run() function.

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
                `Tensor, Tensor, Operation, eval_metrics=dict<str, Tensor>`
                (predictions, loss, train_op, and eval_metrics (if any)),
                `ModelFnOps`.
        input_fn: A function to serve input Tensors fed into the model.
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

    Args:
        name: Flag name. Real flag name will be `"--{}".format(name)`.
        *args, **kwargs: The rest arguments are the same as
            `argparse.add_argument()`.


add_required_flag(name, *args, **kwargs)
    Add a required flag.

    Its interface is the same as `add_flag()` but `required=True` is set by
    default.
```


## Examples

See [examples](examples) directory.


## License

[The Unlicense](https://unlicense.org)
