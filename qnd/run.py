import tensorflow as tf

from .experiment import def_def_experiment_fn
from .flag import add_flag



def def_run():
  add_flag("output_dir",
           default="output",
           help="Directory for checkpoint and event files")

  def_experiment_fn = def_def_experiment_fn()

  def run(model_fn, input_fn):
    tf.contrib.learn.learn_runner.run(def_experiment_fn(model_fn, input_fn),
                                      ARGS.output_dir)

  return run
