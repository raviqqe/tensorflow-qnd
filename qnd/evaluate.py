from .estimator import def_estimator
from .flag import FLAGS, add_output_dir_flag
from .inputs import def_def_infer_input_fn


def def_evaluate(batch_inputs=True, prepare_filename_queues=True):
    """Define `evaluate()` function.

    See also `help(def_evaluate())`.

    - Args
        - `batch_inputs`: Same as `def_train_and_evaluate()`'s.
        - `prepare_filename_queues`: Same as `def_train_and_evaluate()`'s.

    - Returns
        - `evaluate()` function.
    """
    add_output_dir_flag()

    estimator = def_estimator(distributed=False)
    def_eval_input_fn = def_def_infer_input_fn(batch_inputs,
                                               prepare_filename_queues)

    def evaluate(model_fn, input_fn):
        """Evaluate a model with data sample fed by `input_fn`.

        - Args
            - `model_fn`: Same as `train_and_evaluate()`'s.
            - `input_fn`: Same as `eval_input_fn` argument of
                `train_and_evaluate()`.

        - Returns
            - Evaluation results. See `Evaluable` interface in TensorFlow.
        """
        return estimator(model_fn, FLAGS.output_dir).evaluate(
            input_fn=def_eval_input_fn(input_fn))

    return evaluate
