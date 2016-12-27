from .estimator import def_estimator
from .flag import FLAGS, add_output_dir_flag
from .inputs import def_def_infer_input_fn


def def_infer(batch_inputs=True, prepare_filename_queues=True):
    """Define `infer()` function.

    See also `help(def_infer())`.

    - Args
        - `batch_inputs`: Same as `def_train_and_evaluate()`'s.
        - `prepare_filename_queues`: Same as `def_train_and_evaluate()`'s.

    - Returns
        - `infer()` function.
    """
    add_output_dir_flag()

    estimator = def_estimator(distributed=False)
    def_infer_input_fn = def_def_infer_input_fn(batch_inputs,
                                                prepare_filename_queues)

    def infer(model_fn, input_fn):
        """Infer labels or regression values from features of samples fed by
        `input_fn`.

        - Args
            - `model_fn`: Same as `train_and_evaluate()`'s.
            - `input_fn`: Same as `train_input_fn` and `eval_input_fn`
                arguments of `train_and_evaluate()` but returns only features.

        - Returns
            - Generator of inferred label(s) or regression value(s) for each
                sample.
        """
        return estimator(model_fn, FLAGS.output_dir).predict(
            input_fn=def_infer_input_fn(input_fn))

    return infer
