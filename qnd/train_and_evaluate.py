import tensorflow.contrib.learn.python.learn.learn_runner as learn_runner

from .experiment import def_def_experiment_fn
from .flag import FLAGS, add_output_dir_flag


def def_train_and_evaluate(batch_inputs=True,
                           prepare_filename_queues=True,
                           standalone=False):
    """Define `train_and_evaluate()` function.

    See also `help(def_train_and_evaluate())`.

    - Args
        - `batch_inputs`: If `True`, create batches from Tensors returned from
            `train_input_fn()` and `eval_input_fn()` and feed them to a model.
        - `prepare_filename_queues`: If `True`, create filename queues for
            train and eval data based on file paths specified by command line
            arguments.

    - Returns
        - `train_and_evaluate()` function.
    """
    add_output_dir_flag()

    def_experiment_fn = def_def_experiment_fn(batch_inputs,
                                              prepare_filename_queues,
                                              standalone)

    def train_and_evaluate(model_fn, train_input_fn, eval_input_fn=None):
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

    return train_and_evaluate
