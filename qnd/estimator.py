import functools
import inspect

import tensorflow as tf

from . import util
from .config import def_config



def def_estimator():
  config = def_config()

  @util.func_scope
  def estimator(model_fn, model_dir):
    # Hyperparameters (values of `params`) are set by command line arguments.
    return tf.contrib.learn.Estimator(_wrap_model_fn(model_fn),
                                      config=config(),
                                      model_dir=model_dir)

  return estimator


def _wrap_model_fn(original_model_fn):
  @util.func_scope
  def wrapped_model(features, targets, mode):
    are_args = functools.partial(util.are_instances, [features, targets])
    def_model_fn = functools.partial(functools.partial, original_model_fn)

    if are_args(tf.Tensor):
      model_fn = def_model_fn(features, targets)
    elif are_args(dict):
      model_fn = def_model_fn(**features, **targets)
    else:
      raise ValueError("features and targets should be both tf.Tensor or "
                       "dict.")

    predictions, loss, train_op, *eval_metric_ops = (
        model_fn(mode=mode)
        if "mode" in inspect.signature(model_fn).parameters.keys() else
        model_fn())

    return tf.contrib.learn.estimators.model_fn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=(eval_metric_ops[0] if eval_metric_ops else None))

  return wrapped_model
