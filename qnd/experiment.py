import tensorflow as tf
from gargparse import ARGS

from . import flag
from .estimator import def_estimator
from .input import def_train_input_fn
from .input import def_eval_input_fn



# TODO: Where to set this? Or, is this the same as Estimator.model_dir?
# add_flag("summary_dir", default="summary", help="Directory containing checkpoint and event files")


def def_experiment():
  adder = flag.FlagAdder()
  adder.add_flag("train_steps", type=int)
  adder.add_flag("eval_steps", type=int, "Works only for qnd.evaluate()")
  adder.add_flag("min_eval_frequency", type=int, default=1,
                 "Works only for qnd.train_and_evaluate()")

  estimator = def_estimator()
  train_input_fn = def_train_input_fn()
  eval_input_fn = def_eval_input_fn()

  def experiment(model_fn, file_reader):
    return tf.contirb.learn.Experiment(
        estimator(model_fn),
        train_input_fn(file_reader),
        eval_input_fn(file_reader),
        **{arg: getattr(ARGS, arg) for arg in adder.flags})

  return experiment
