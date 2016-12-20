import functools

import tensorflow as tf
from gargparse import ARGS

from . import util
from .config import def_config
from .flag import add_flag



def def_estimator():
  add_flag("model_dir", default="model", help="Directory for checkpoint files")
  config = def_config()

  @util.func_scope
  def estimator(model_fn):
    # Hyperparameters (values of `params`) are set by command line arguments.
    return tf.contrib.learn.Estimator(_wrap_model_fn(model_fn),
                                      config=config(),
                                      model_dir=ARGS.model_dir)

  return estimator


def _wrap_model_fn(model_fn):
  @util.func_scope
  def wrapped_model(features, targets, mode):
    model_fn = functools.partial(model_fn, **features, **targets)

    predictions, loss, train_op, *eval_metric_ops = (
        model_fn(mode=mode)
        if "mode" in inspect.signature(model_fn).keys() else
        model_fn())

    return tf.contirb.learn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=(eval_metric_ops[0] if eval_metric_ops else None))

  return wrapped_model
