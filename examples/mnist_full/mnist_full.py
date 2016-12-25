import logging
import os
import sys

import qnd
import tensorflow as tf


BATCH_SIZE = 64


def env(name):
    assert name in ["use_eval_input_fn", "use_dict_inputs", "use_model_fn_ops",
                    "self_batch"]

    env_is_set = name in os.environ

    if env_is_set:
        print("Environment variable, {} is set.".format(name), file=sys.stderr)

    return env_is_set


def train_batch(input_fn):
    def batched_input_fn(filename_queue):
        capacity = BATCH_SIZE * 10
        return tf.train.shuffle_batch(input_fn(filename_queue),
                                      batch_size=BATCH_SIZE,
                                      capacity=capacity,
                                      min_after_dequeue=capacity // 2)

    return batched_input_fn


def eval_batch(input_fn):
    def batched_input_fn(filename_queue):
        return tf.train.batch(input_fn(filename_queue), batch_size=BATCH_SIZE)

    return batched_input_fn


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


run = qnd.def_run(batch_inputs=(not env("self_batch")))


def main():
    logging.getLogger().setLevel(logging.INFO)

    def input_fn(batch_fn):
        return batch_fn(read_file) if env("self_batch") else read_file

    run(model,
        input_fn(train_batch),
        input_fn(eval_batch) if env("use_eval_input_fn") else None)


if __name__ == "__main__":
    main()
