import functools
import json
import logging
import os

import tensorflow as tf

from . import util
from . import flag
from .flag import FLAGS, add_flag, add_required_flag


_JOBS = {getattr(tf.contrib.learn.TaskType, name)
         for name in ["MASTER", "PS", "WORKER"]}


def def_config():
    # ClusterConfig flags

    add_required_flag("master_host",
                      help="$hostname:$port pair of a master host")

    def add_hosts_flag(name, **kwargs):
        return add_flag(
            name,
            type=(lambda string: string.split(',')),
            default=[],
            help="Comma-separated list of $hostname:$port pairs of {}"
            .format(name.replace("_", " ")),
            **kwargs)

    add_hosts_flag("ps_hosts", required=True)
    add_hosts_flag("worker_hosts")

    add_required_flag(
        "task_type", help="Must be in {} (aka job)".format(_JOBS))
    add_flag("task_index", type=int, default=0, help="Task index within a job")

    # RunConfig flags

    adder = flag.FlagAdder()
    # Default values are based on ones of tf.contrib.learn.RunConfig.
    adder.add_flag("num_cores", type=int, default=0,
                   help="Number of CPU cores used. "
                        "0 means use of a default value.")
    adder.add_flag("log_device_placement", action="store_true",
                   help="If specified, log device placement information")

    def saver_help(x):
        return "Number of steps every time of which {} is saved" \
               .format(x)

    adder.add_flag("save_summary_steps", type=int, default=100,
                   help=saver_help("summary"))
    adder.add_flag("save_checkpoints_steps", type=int,
                   help=saver_help("a model"))

    @util.func_scope
    def config():
        config_env = "TF_CONFIG"

        if config_env in os.environ and os.environ[config_env]:
            logging.warning("A value of the environment variable of TensorFlow "
                            "cluster configuration, {} is discarded."
                            .format(config_env))

        if FLAGS.master_host in FLAGS.worker_hosts:
            raise ValueError("Master host {} is found in worker hosts {}."
                             .format(FLAGS.master_host, FLAGS.worker_hosts))

        if FLAGS.task_type not in _JOBS:
            raise ValueError("Specified task type (job) {} is not in available "
                             "task types {}".format(FLAGS.task_type, _JOBS))

        os.environ[config_env] = json.dumps({
            "environment": tf.contrib.learn.Environment.CLOUD,
            "cluster": {
                "master": [FLAGS.master_host],
                "ps": FLAGS.ps_hosts,
                "worker": [FLAGS.master_host, *FLAGS.worker_hosts],
            },
            "task": {
                "type": FLAGS.task_type,
                "index": FLAGS.task_index,
            },
        })

        return tf.contrib.learn.RunConfig(**adder.flags)

    return config
