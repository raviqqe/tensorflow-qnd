import tensorflow.contrib.learn.python.learn.learn_runner as learn_runner

from .experiment import def_def_experiment_fn
from .flag import FLAGS, add_flag


def def_run(batch_inputs=True, prepare_filename_queues=True):
    """Define `run()` function.

    See also `help(def_run())`.

    - Args
        - `batch_inputs`: If `True`, create batches from Tensors returned from
            `train_input_fn()` and `eval_input_fn()` and feed them to a model.
        - `prepare_filename_queues`: If `True`, create filename queues for
            train and eval data based on file paths specified by command line
            arguments.

    - Returns
        - `run()` function.
    """
    add_flag("output_dir",
             default="output",
             help="Directory where checkpoint and event files are stored")

    def_experiment_fn = def_def_experiment_fn(batch_inputs,
                                              prepare_filename_queues)

    def run(model_fn, train_input_fn, eval_input_fn=None):
        """Run `tf.contrib.learn.python.learn.learn_runner.run()`.

        - Args
            - `model_fn`: A function to construct a model.
                - Types of its arguments must be one of the following:
                    - `Tensor, ...`
                    - `Tensor, ..., mode=ModeKeys`
                - Types of its return values must be one of the following:
                    - `Tensor, Tensor, Operation, eval_metric_ops=dict<str, Tensor>`
                        (predictions, loss, train_op, and eval_metric_ops (if any))
                    - `ModelFnOps`
            - `train_input_fn`, `eval_input_fn`: Functions to serve input
                Tensors fed into the model. If `eval_input_fn` is `None`,
                `train_input_fn` will be used instead.
                - Types of its arguments must be one of the following:
                    - `QueueBase` (a filename queue)
                    - `None` (No argument)
                - Types of its return values must be one of the following:
                    - `Tensor, Tensor` (x and y)
                    - `dict<str, Tensor>, dict<str, Tensor>` (features and labels)
                        - The keys in `dict` objects must match with names of
                            `model_fn`'s arguments.

        - Returns
            - Return value of `tf.contrib.learn.python.learn.learn_runner.run()`.
        """
        return learn_runner.run(
            def_experiment_fn(model_fn, train_input_fn, eval_input_fn),
            FLAGS.output_dir)

    return run
