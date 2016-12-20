import functools
import json
import logging
import os

import tensorflow as tf
from gargparse import ARGS

from . import util
from . import flag
from .flag import add_required_flag



def def_config():
  # ClusterConfig flags

  add_hosts_flag = functools.partial(
      add_required_flag,
      type=flag.str_list,
      help="Comma-separated list of hostname:port pairs")
  add_hosts_flag("ps_hosts")
  add_hosts_flag("worker_hosts")

  add_required_flag("task_type", help="'ps' or 'worker' (aka job)")
  add_required_flag("task_index", type=int)

  # RunConfig flags

  adder = flag.FlagAdder()
  # Default values are based on ones of tf.contrib.learn.RunConfig.
  adder.add_flag("num_cores", type=int, default=0)
  adder.add_flag("log_device_placement", action="store_true")
  adder.add_flag("save_summary_steps", type=int, default=100)
  adder.add_flag("save_checkpoints_steps", type=int)

  @util.func_scope
  def config():
    config_env = "TF_CONFIG"

    if config_env in os.environ and os.environ[config_env]:
      logging.warning("A value of the environment variable of TensorFlow "
                      "cluster configuration, {} is discarded."
                      .format(config_env))

    os.environ[config_env] = json.dumps({
      "cluster": {
        "ps": ARGS.ps_hosts,
        "worker": ARGS.worker_hosts,
      },
      "task_id": {
        "type": ARGS.task_type,
        "index": ARGS.task_index,
      },
    })

    return tf.contrib.learn.RunConfig(
        **{arg: getattr(ARGS, arg) for arg in adder.flags})

  return config
