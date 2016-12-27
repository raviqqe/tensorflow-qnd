from .estimator import def_estimator
from .flag import FLAGS, add_flag
from .inputs import def_def_eval_input_fn


def def_infer(batch_inputs=True, prepare_filename_queues=True):
    add_flag("output_dir",
             default="output",
             help="Directory where checkpoint and event files are stored")

    estimator = def_estimator()
    def_eval_input_fn = def_def_eval_input_fn(batch_inputs,
                                              prepare_filename_queues)

    def infer(model_fn, input_fn):
        return estimator(model_fn, FLAGS.output_dir).predict(
            input_fn=def_eval_input_fn(input_fn))

    return infer
