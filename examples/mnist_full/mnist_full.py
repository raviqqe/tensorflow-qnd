import logging
import os

import qnd
import tensorflow as tf

import mnist


BATCH_SIZE = 64
ENV_FLAGS = ["use_eval_input_fn", "use_dict_inputs", "use_model_fn_ops",
             "self_batch", "self_filename_queue"]


def env(name):
    assert name in ENV_FLAGS
    return name in os.environ


for name in ENV_FLAGS:
    if env(name):
        print("Environment variable, {} is set.".format(name))

if env("self_filename_queue"):
    qnd.add_required_flag("train_file")
    qnd.add_required_flag("eval_file")


def filename_queue(train=False):
    train_filenames = tf.train.match_filenames_once(
        qnd.FLAGS.train_file, name="train_filenames")
    eval_filenames = tf.train.match_filenames_once(
        qnd.FLAGS.eval_file, name="eval_filenames")

    return tf.train.string_input_producer(
        train_filenames if train else eval_filenames,
        num_epochs=(None if train else 1),
        shuffle=train)


def train_batch(*tensors):
    capacity = BATCH_SIZE * 10
    return tf.train.shuffle_batch(tensors,
                                  batch_size=BATCH_SIZE,
                                  capacity=capacity,
                                  min_after_dequeue=capacity // 2)


def eval_batch(*tensors):
    return tf.train.batch(tensors, batch_size=BATCH_SIZE)


def read_file(filename_queue):
    image, number = mnist.read_file(filename_queue)

    return (({"image": image}, {"number": number})
            if env("use_dict_inputs") else
            (image, number))


mnist_model = mnist.def_model()


def model(image, number=None, mode=tf.contrib.learn.ModeKeys.TRAIN):
    results = mnist_model(image, number, mode)

    return (tf.contrib.learn.estimators.model_fn.ModelFnOps(mode, *results)
            if env("use_model_fn_ops") else
            results)


train_and_evaluate = qnd.def_train_and_evaluate(
    batch_inputs=(not env("self_batch")),
    prepare_filename_queues=(not env("self_filename_queue")))


def main():
    logging.getLogger().setLevel(logging.INFO)

    def def_input_fn(batch_fn, filename_queue_fn):
        def batch(*tensors):
            return batch_fn(*tensors) if env("self_batch") else tensors

        if env("self_filename_queue"):
            def input_fn():
                return batch(*read_file(filename_queue_fn()))
        else:
            def input_fn(filename_queue):
                return batch(*read_file(filename_queue))

        return input_fn

    train_and_evaluate(
        model,
        def_input_fn(train_batch, lambda: filename_queue(train=True)),
        (def_input_fn(eval_batch, lambda: filename_queue())
         if env("use_eval_input_fn") else
         None))


if __name__ == "__main__":
    main()
