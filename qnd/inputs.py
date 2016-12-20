import tensorflow as tf
import gargparse
from gargparse import ARGS

from . import util
from .flag import add_flag, add_required_flag



def add_file_flag(use):
  flag_name = "{}_file".format(use)
  add_required_flag(flag_name,
                    help="File path of {} data file(s). "
                         "Glob is accepted. (e.g. train/*.csv)".format(use))
  return flag_name


def def_def_input_fn(use):
  def def_input_fn():
    file_flag = add_file_flag(use)
    read_files = def_read_files()

    @util.func_scope
    def input_fn(user_input_fn):
      return read_files(getattr(ARGS, file_flag), user_input_fn)

    return input_fn

  return def_input_fn


def_train_input_fn = def_def_input_fn("train")
def_eval_input_fn = def_def_input_fn("eval")


def def_read_files():
  file_pattern_to_name_queue = def_file_pattern_to_name_queue()

  @util.func_scope
  def read_files(file_pattern, user_input_fn):
    return user_input_fn(file_pattern_to_name_queue(file_pattern))

  return read_files


def def_file_pattern_to_name_queue():
  add_flag("num_epochs", type=int)
  add_flag("filename_queue_capacity", type=int, default=32)

  @util.func_scope
  def file_pattern_to_name_queue(pattern):
    return tf.train.string_input_producer(
        tf.train.match_filenames_once(pattern),
        num_epochs=ARGS.num_epochs,
        capacity=ARGS.filename_queue_capacity)

  return file_pattern_to_name_queue
