import enum

import tensorflow as tf

from . import util
from .flag import FLAGS, add_flag, add_required_flag


class DataUse(enum.Enum):
    TRAIN = "train"
    EVAL = "eval"


def add_file_flag(use):
    assert isinstance(use, str)

    flag_name = "{}_file".format(use)
    add_required_flag(flag_name,
                      help="File path of {0} data file(s). "
                           "A glob is available. (e.g. {0}/*.tfrecords)"
                           .format(use))
    return flag_name


def def_def_def_input_fn(use):
    assert isinstance(use, DataUse)

    def def_def_input_fn():
        file_flag = add_file_flag(use.value)
        read_files = def_read_files(use)

        def def_input_fn(user_input_fn):
            @util.func_scope
            def input_fn():
                return read_files(getattr(FLAGS, file_flag), user_input_fn)

            return input_fn

        return def_input_fn

    return def_def_input_fn


def_def_train_input_fn = def_def_def_input_fn(DataUse.TRAIN)
def_def_eval_input_fn = def_def_def_input_fn(DataUse.EVAL)


def def_read_files(use):
    file_pattern_to_name_queue = def_file_pattern_to_name_queue(use)

    @util.func_scope
    def read_files(file_pattern, user_input_fn):
        return user_input_fn(file_pattern_to_name_queue(file_pattern))

    return read_files


def def_file_pattern_to_name_queue(use):
    assert isinstance(use, DataUse)

    if use == DataUse.TRAIN:
        add_flag("num_epochs", type=int, help="Number of epochs")
    add_flag("filename_queue_capacity", type=int, default=32,
             help="Capacity of filename queues of {} and {} data"
                  .format(*[getattr(DataUse, use).value
                            for use in ["TRAIN", "EVAL"]]))

    @util.func_scope
    def file_pattern_to_name_queue(pattern):
        return tf.train.string_input_producer(
            tf.train.match_filenames_once(pattern),
            num_epochs=(FLAGS.num_epochs if use == DataUse.TRAIN else 1),
            capacity=FLAGS.filename_queue_capacity)

    return file_pattern_to_name_queue
