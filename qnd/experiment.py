import tensorflow as tf

from .flag import FLAGS, FlagAdder
from .estimator import def_estimator
from .inputs import DataUse, def_def_train_input_fn, def_def_eval_input_fn


def def_def_experiment_fn():
    adder = FlagAdder()

    for use in DataUse:
        use = use.value
        adder.add_flag("{}_steps".format(use), type=int,
                       help="Maximum number of {} steps".format(use))

    adder.add_flag("min_eval_frequency", type=int, default=1,
                   help="Minimum evaluation frequency in number of model "
                        "savings")

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
