import functools
import inspect
import typing

import tensorflow as tf
import tensorflow.contrib.learn as learn

from . import util
from .config import def_config


def def_estimator(distributed=False):
    config = def_config(distributed)

    @util.func_scope
    def estimator(model_fn, model_dir):
        return tf.contrib.learn.Estimator(_wrap_model_fn(model_fn),
                                          config=config(),
                                          model_dir=model_dir)

    return estimator


def _wrap_model_fn(original_model_fn):
    @util.func_scope
    def model(features, targets, mode):
        are_args = functools.partial(util.are_instances, [features, targets])
        def_model_fn = functools.partial(functools.partial, original_model_fn)

        if are_args(tf.Tensor):
            model_fn = def_model_fn(features, targets)
        elif are_args(dict):
            model_fn = def_model_fn(**features, **targets)
        elif isinstance(features, tf.Tensor) and targets is None:
            model_fn = def_model_fn(features)
        elif isinstance(features, dict) and targets is None:
            model_fn = def_model_fn(**features)
        else:
            raise ValueError(
                "features and targets should be both tf.Tensor or dict.")

        results = (
            model_fn(mode=mode)
            if "mode" in inspect.signature(model_fn).parameters.keys() else
            model_fn())

        return (
            results
            if isinstance(results, learn.estimators.model_fn.ModelFnOps) else
            learn.estimators.model_fn.ModelFnOps(
                mode,
                *(results
                  if isinstance(results, typing.Sequence) else
                  (results,))))

    return model
