import functools
import json
import logging
import os

import tensorflow as tf
import gargparse
from gargparse import ARGS

from . import util



def add_file_flag(use):
  flag_name = "{}_file".format(use)
  add_required_flag(flag_name,
                    help="File path of {} dataset file(s). "
                         "Glob is accepted. (e.g. eval/*.csv)".format(use))
  return flag_name


def def_def_input_fn(use):
  def def_input_fn(file_parser):
    file_flag = add_file_flag(use)
    read_files = def_read_files()

    @util.func_scope
    def input_fn():
      return read_files(getattr(ARGS, file_flag), file_parser)

    return input_fn

  return def_input_fn


def_train_input_fn = def_def_input_fn("train")
def_eval_input_fn = def_def_input_fn("eval")


def def_read_files():
  add_flag("batch_queue_capacity", type=int, default=64)
  file_pattern_to_name = def_file_pattern_to_name()

  @util.func_scope
  def read_files(file_pattern, file_parser):
    return monitored_queue(
        *file_parser(file_pattern_to_name(file_pattern)),
        queue_size_name="batches_in_queue",
        capacity=ARGS.batch_queue_capacity)

  return read_files


def def_file_pattern_to_name():
  add_flag("num_epochs", type=int)
  add_flag("filename_queue_capacity", type=int, default=32)

  @util.func_scope
  def file_pattern_to_name(pattern):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(pattern),
        num_epochs=ARGS.num_epochs,
        capacity=FLAGS.filename_queue_capacity)

    tf.summary.scalar("filenames_in_queue", filename_queue.size())

    return filename_queue.dequeue()

  return file_pattern_to_names


def def_monitored_queue():
  @util.func_scope
  def monitored_queue(*tensors,
                      capacity,
                      *,
                      queue_size_name,
                      return_queue=False):
    queue = tf.FIFOQueue(capacity, dtypes(*tensors))
    tf.summary.scalar(summary_name, queue.size())
    tf.train.add_queue_runner(queue, [queue.enqueue(tensors)])

    if return_queue:
      return queue

    results = queue.dequeue()

    for tensor, result \
        in zip(tensors, results if isinstance(results, list) else [results]):
      result.set_shape(tensor.get_shape())

    return results

  return monitored_queue
