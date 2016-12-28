import enum

import tensorflow as tf

from . import util
from .flag import FLAGS, add_flag, add_required_flag


MODES = [tf.contrib.learn.ModeKeys.TRAIN,
         tf.contrib.learn.ModeKeys.EVAL,
         tf.contrib.learn.ModeKeys.INFER]


def _add_file_flag(mode):
    assert isinstance(mode, str)

    flag_name = "{}_file".format(mode)
    add_required_flag(flag_name,
                      help="File path of {0} data file(s). "
                           "A glob is available. (e.g. {0}/*.tfrecords)"
                           .format(mode))
    return flag_name


def def_def_def_input_fn(mode):
    assert mode in MODES

    BATCH_SIZE = 64

    def def_def_input_fn(batch_inputs=True, prepare_filename_queues=True):
        if batch_inputs:
            add_flag("batch_size", type=int, default=BATCH_SIZE,
                     help="Mini-batch size")
            add_flag("batch_queue_capacity", type=int, default=BATCH_SIZE * 16,
                     help="Batch queue capacity")

        if prepare_filename_queues:
            file_flag = _add_file_flag(mode)
            filenames_to_queue = def_filenames_to_queue(mode)

        def def_input_fn(user_input_fn):
            @util.func_scope
            def input_fn():
                inputs = (
                    user_input_fn(filenames_to_queue(
                        tf.matching_files(FLAGS.infer_file)
                        if mode == tf.contrib.learn.ModeKeys.INFER else
                        {mode: tf.train.match_filenames_once(
                            getattr(FLAGS, "{}_file".format(mode)),
                            name="{}_filenames".format(mode))
                         for mode in [tf.contrib.learn.ModeKeys.TRAIN,
                                      tf.contrib.learn.ModeKeys.EVAL]}[mode]))
                    if prepare_filename_queues else
                    user_input_fn())

                inputs = ((inputs,)
                          if type(inputs) in {dict, tf.Tensor} else
                          inputs)

                _check_inputs(inputs)

                return _batch_inputs(inputs, mode) if batch_inputs else inputs

            return input_fn

        return def_input_fn

    return def_def_input_fn


def _batch_inputs(inputs, mode):
    tensor_input = isinstance(inputs[0], tf.Tensor)

    batched_inputs = (tf.train.shuffle_batch
                      if mode == tf.contrib.learn.ModeKeys.TRAIN else
                      tf.train.batch)(
        [*inputs] if tensor_input else _merge_dicts(*inputs),
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.batch_queue_capacity,
        **({"min_after_dequeue": FLAGS.batch_queue_capacity // 2}
           if mode == tf.contrib.learn.ModeKeys.TRAIN else
           {"allow_smaller_final_batch": True}))

    restore_dict = lambda x: {key: batched_inputs[key] for key in x.keys()}

    return batched_inputs if tensor_input else [*map(restore_dict, inputs)]


def _merge_dicts(*dicts):
    return {key: value for dict_ in dicts for key, value in dict_.items()}


def _check_inputs(inputs):
    if len(inputs) not in {1, 2}:
        raise ValueError("Too many return values from input_fn. "
                         "(returned values: {})"
                         .format(inputs))

    if len(inputs) == 2 and type(inputs[0]) != type(inputs[1]):
        raise ValueError("features and targets should be the same type. "
                         "(features type: {}, targets type: {})"
                         .format(*map(type, inputs)))

    if len(inputs) == 2 and isinstance(inputs[0], dict):
        duplicate_keys = inputs[0].keys() & inputs[1].keys()
        if len(duplicate_keys) != 0:
            raise ValueError(
                "Some keys of features and targets are duplicate. ({})"
                .format(duplicate_keys))


for mode in MODES:
    globals()["def_def_{}_input_fn".format(mode)] = def_def_def_input_fn(mode)


def def_filenames_to_queue(mode):
    assert mode in MODES

    add_flag("filename_queue_capacity", type=int, default=32,
             help="Capacity of filename queues of {}, {} and {} data"
                  .format(*MODES))

    @util.func_scope
    def filenames_to_queue(filenames):
        return tf.train.string_input_producer(
            filenames,
            num_epochs=(None
                        if mode == tf.contrib.learn.ModeKeys.TRAIN else
                        1),
            shuffle=(mode == tf.contrib.learn.ModeKeys.TRAIN),
            capacity=FLAGS.filename_queue_capacity)

    return filenames_to_queue
