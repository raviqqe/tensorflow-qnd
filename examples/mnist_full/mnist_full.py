import logging
import os

import qnd
import tensorflow as tf


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
    _, serialized = tf.TFRecordReader().read(filename_queue)

    scalar_feature = lambda dtype: tf.FixedLenFeature([], dtype)

    features = tf.parse_single_example(serialized, {
        "image_raw": scalar_feature(tf.string),
        "label": scalar_feature(tf.int64),
    })

    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image.set_shape([28**2])
    image = tf.to_float(image) / 255 - 0.5
    number = features["label"]

    return (({"image": image}, {"number": number})
            if env("use_dict_inputs") else
            (image, number))


def minimize(loss):
    return tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        0.001,
        "Adam")


def model(image, number, mode):
    h = tf.contrib.layers.fully_connected(image, 64)
    h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(h, number))
    predictions = tf.argmax(h, axis=1)
    train_op = minimize(loss)
    eval_metric_ops = {
        "accuracy": tf.contrib.metrics.streaming_accuracy(predictions,
                                                          number)[1],
    }

    if env("use_model_fn_ops"):
        return tf.contrib.learn.estimators.model_fn.ModelFnOps(
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            mode=mode)

    return predictions, loss, train_op, eval_metric_ops


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
