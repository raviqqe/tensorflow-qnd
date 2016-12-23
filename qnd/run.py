import tensorflow.contrib.learn.python.learn.learn_runner as learn_runner

from .experiment import def_def_experiment_fn
from .flag import FLAGS, add_flag


def def_run(batch_inputs=True, prepare_filename_queues=True):
    add_flag("output_dir",
             default="output",
             help="Directory where checkpoint and event files are stored")

    def_experiment_fn = def_def_experiment_fn(batch_inputs,
                                              prepare_filename_queues)

    def run(model_fn, train_input_fn, eval_input_fn=None):
        return learn_runner.run(
            def_experiment_fn(model_fn, train_input_fn, eval_input_fn),
            FLAGS.output_dir)

    return run
