import tensorflow.contrib.learn.python.learn.learn_runner as learn_runner

from .experiment import def_def_experiment_fn
from .flag import FLAGS, add_output_dir_flag


def def_train_and_evaluate(batch_inputs=True,
                           prepare_filename_queues=True,
                           distributed=False):
    """Define `train_and_evaluate()` function.

    See also `help(def_train_and_evaluate())`.

    - Args
        - `batch_inputs`: If `True`, create batches from Tensors returned from
            `train_input_fn()` and `eval_input_fn()` and feed them to a model.
        - `prepare_filename_queues`: If `True`, create filename queues for
            train and eval data based on file paths specified by command line
            arguments.
        - `distributed`: If `True`, configure command line arguments to train
            and evaluate models on a distributed system.

    - Returns
        - `train_and_evaluate()` function.
    """
    add_output_dir_flag()

    def_experiment_fn = def_def_experiment_fn(batch_inputs,
                                              prepare_filename_queues,
                                              distributed)

    def train_and_evaluate(model_fn, train_input_fn, eval_input_fn=None):
        """Train and evaluate a model with features and targets fed by
        `input_fn`s.

        - Args
            - `model_fn`: A function to construct a model.
                - Types of its arguments must be one of the following:
                    - `Tensor, ...`
                    - `Tensor, ..., mode=ModeKeys`
                - Types of its return values must be one of the following:
                    - `Tensor, Tensor, Operation, eval_metric_ops=dict<str, Tensor>`
                        (predictions, loss, train_op, and eval_metric_ops (if any))
                    - `ModelFnOps`
            - `train_input_fn`, `eval_input_fn`: Functions to create input
                Tensors fed into the model. If `eval_input_fn` is `None`,
                `train_input_fn` will be used instead.
                - Types of its arguments must be one of the following:
                    - `QueueBase` (a filename queue)
                    - No argument if `prepare_filename_queues` of
                        `def_train_and_evaluate()` is `False`.
                - Types of its return values must be one of the following:
                    - `Tensor, Tensor` (features and targets)
                    - `dict<str, Tensor>, dict<str, Tensor>` (features and targets)
                        - The keys in `dict` objects must match with argument
                            names of `model_fn`.

        - Returns
            - Return value of `tf.contrib.learn.python.learn.learn_runner.run()`.
        """
        return learn_runner.run(
            def_experiment_fn(model_fn, train_input_fn, eval_input_fn),
            FLAGS.output_dir)

    return train_and_evaluate
