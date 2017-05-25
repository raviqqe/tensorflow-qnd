import os

import tensorflow as tf

from . import util
from .flag import FLAGS, add_flag, add_required_flag


DEFAULT_BATCH_SIZE = 64
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

    def def_def_input_fn(batch_inputs=True, prepare_filename_queues=True):
        if batch_inputs:
            add_flag("batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                     help="Mini-batch size")
            add_flag("batch_queue_capacity",
                     type=int,
                     # TODO: Set a value for predictable behavior
                     default=DEFAULT_BATCH_SIZE * 16,
                     help="Batch queue capacity")
            add_flag("num_batch_threads", type=int, default=os.cpu_count(),
                     help="Number of threads used to create batches")

        if prepare_filename_queues:
            _add_file_flag(mode)
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

                inputs = ([inputs]
                          if type(inputs) in {dict, tf.Tensor} else
                          inputs)

                _check_inputs(inputs)

                return _batch_inputs(inputs, mode) if batch_inputs else inputs

            return input_fn

        return def_input_fn

    return def_def_input_fn


def _batch_inputs(inputs, mode):
    input_is_dict = isinstance(inputs[0], dict)

    batched_inputs = _batch_merged_inputs(
        _merge_dicts(*inputs) if input_is_dict else inputs,
        mode)

    return ([{key: batched_inputs[key] for key in input_.keys()}
             for input_ in inputs]
            if input_is_dict else
            batched_inputs)


def _batch_merged_inputs(inputs, mode):
    if mode != tf.contrib.learn.ModeKeys.INFER:
        inputs = _shuffle(inputs,
                          capacity=FLAGS.batch_queue_capacity,
                          num_threads=FLAGS.num_batch_threads,
                          # TODO: Set a proper value for predictable behavior
                          min_after_dequeue=FLAGS.batch_queue_capacity // 2)

    return tf.train.batch(
        inputs,
        batch_size=FLAGS.batch_size,
        dynamic_pad=True,
        capacity=FLAGS.batch_queue_capacity,
        num_threads=FLAGS.num_batch_threads,
        allow_smaller_final_batch=(mode != tf.contrib.learn.ModeKeys.TRAIN))


def _shuffle(inputs, capacity, min_after_dequeue, num_threads):
    if isinstance(inputs, dict):
        names, dtypes = zip(*[(key, input_.dtype)
                              for key, input_ in inputs.items()])
    else:
        dtypes = [input_.dtype for input_ in inputs]

    queue = tf.RandomShuffleQueue(
        capacity,
        min_after_dequeue,
        dtypes,
        **({'names': names} if isinstance(inputs, dict) else {}))

    tf.train.add_queue_runner(tf.train.QueueRunner(
        queue,
        [queue.enqueue(inputs)] * num_threads))

    shuffled_inputs = queue.dequeue()

    for key, input_ in (inputs.items()
                        if isinstance(inputs, dict) else
                        enumerate(inputs)):
        shuffled_inputs[key].set_shape(input_.get_shape())

    return shuffled_inputs


def _merge_dicts(*dicts):
    return {key: value for dict_ in dicts for key, value in dict_.items()}


def _check_inputs(inputs):
    if len(inputs) not in {1, 2}:
        raise ValueError("Too many return values from input_fn. "
                         "(returned values: {})"
                         .format(inputs))

    if len(inputs) == 2 and not isinstance(inputs[0], type(inputs[1])):
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
            shuffle=(mode != tf.contrib.learn.ModeKeys.INFER),
            capacity=FLAGS.filename_queue_capacity)

    return filenames_to_queue
