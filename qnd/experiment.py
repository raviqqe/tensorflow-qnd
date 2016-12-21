import tensorflow as tf

from .flag import FLAGS, FlagAdder
from .estimator import def_estimator
from .inputs import def_def_train_input_fn
from .inputs import def_def_eval_input_fn



def def_def_experiment_fn():
  adder = FlagAdder()

  works_with = lambda name: "Works only with {}".format(name)
  train_help = works_with("qnd.train_and_evaluate()")

  adder.add_flag("train_steps", type=int, help=train_help)
  adder.add_flag("eval_steps", type=int, help=works_with("qnd.evaluate()"))
  adder.add_flag("min_eval_frequency", type=int, default=1, help=train_help)

  estimator = def_estimator()
  def_train_input_fn = def_def_train_input_fn()
  def_eval_input_fn = def_def_eval_input_fn()

  def def_experiment_fn(model_fn, input_fn):
    def experiment_fn(output_dir):
      return tf.contrib.learn.Experiment(
          estimator(model_fn, output_dir),
          def_train_input_fn(input_fn),
          def_eval_input_fn(input_fn),
          **{arg: getattr(FLAGS, arg) for arg in adder.flags})

    return experiment_fn

  return def_experiment_fn
