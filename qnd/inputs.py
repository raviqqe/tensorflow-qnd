import enum

import tensorflow as tf

from . import util
from .flag import FLAGS, add_flag, add_required_flag


class Mode(enum.Enum):
    TRAIN = "train"
    EVAL = "eval"


def _add_file_flag(mode):
    assert isinstance(mode, str)

    flag_name = "{}_file".format(mode)
    add_required_flag(flag_name,
                      help="File path of {0} data file(s). "
                           "A glob is available. (e.g. {0}/*.tfrecords)"
                           .format(mode))
    return flag_name


def def_def_def_input_fn(mode):
    assert isinstance(mode, Mode)

    BATCH_SIZE = 64

    def def_def_input_fn(batch_inputs=True, prepare_filename_queues=True):
        if batch_inputs:
            add_flag("batch_size", type=int, default=BATCH_SIZE,
                     help="Mini-batch size")
            add_flag("batch_queue_capacity", type=int, default=BATCH_SIZE * 16,
                     help="Batch queue capacity")

        if prepare_filename_queues:
            file_flag = _add_file_flag(mode.value)
            read_files = def_read_files(mode)

        def def_input_fn(user_input_fn):
            @util.func_scope
            def input_fn():
                if prepare_filename_queues:
                    x, y = user_input_fn(read_files(getattr(FLAGS, file_flag)))
                else:
                    x, y = user_input_fn()

                if not batch_inputs:
                    return x, y

                tuple_input = isinstance(x, tf.Tensor)

                if not tuple_input:
                    duplicate_keys = x.keys() & y.keys()
                    if len(duplicate_keys) != 0:
                        raise ValueError(
                            "Some keys of x and y are duplicate. ({})"
                            .format(duplicate_keys))

                inputs = (tf.train.shuffle_batch if mode == Mode.TRAIN else
                          tf.train.batch)(
                    [x, y] if tuple_input else {**x, **y},
                    batch_size=FLAGS.batch_size,
                    capacity=FLAGS.batch_queue_capacity,
                    **({"min_after_dequeue": FLAGS.batch_queue_capacity // 2}
                       if mode == Mode.TRAIN else
                       {"allow_smaller_final_batch": True}))

                restore = lambda x: {key: inputs[key] for key in x.keys()}

                return inputs if tuple_input else (restore(x), restore(y))

            return input_fn

        return def_input_fn

    return def_def_input_fn


def_def_train_input_fn = def_def_def_input_fn(Mode.TRAIN)
def_def_eval_input_fn = def_def_def_input_fn(Mode.EVAL)


def def_read_files(mode):
    file_pattern_to_name_queue = def_file_pattern_to_name_queue(mode)

    @util.func_scope
    def read_files(file_pattern):
        return file_pattern_to_name_queue(file_pattern)

    return read_files


def def_file_pattern_to_name_queue(mode):
    assert isinstance(mode, Mode)

    add_flag("filename_queue_capacity", type=int, default=32,
             help="Capacity of filename queues of {} and {} data"
                  .format(*[mode.value for mode in Mode]))

    @util.func_scope
    def file_pattern_to_name_queue(pattern):
        return tf.train.string_input_producer(
            tf.train.match_filenames_once(pattern),
            num_epochs=(None if mode == Mode.TRAIN else 1),
            shuffle=(mode == Mode.TRAIN),
            capacity=FLAGS.filename_queue_capacity)

    return file_pattern_to_name_queue
