import logging

import qnd
import tensorflow as tf


logging.getLogger().setLevel(logging.INFO)

qnd.add_flag("batch_size", type=int, default=64)
qnd.add_flag("batch_queue_capacity", type=int, default=1024)
qnd.add_flag("use_eval_input_fn",
             type=(lambda string: bool(int(string))),
             default=False)


def read_file(filename_queue):
    _, serialized = tf.TFRecordReader().read(filename_queue)

    scalar_feature = lambda dtype: tf.FixedLenFeature([], dtype)

    features = tf.parse_single_example(serialized, {
        "image_raw": scalar_feature(tf.string),
        "label": scalar_feature(tf.int64),
    })

    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image.set_shape([28**2])
    return [tf.to_float(image) / 255 - 0.5, features["label"]]


def train_input(filename_queue):
    return tf.train.shuffle_batch(
        read_file(filename_queue),
        batch_size=qnd.FLAGS.batch_size,
        capacity=qnd.FLAGS.batch_queue_capacity,
        min_after_dequeue=qnd.FLAGS.batch_queue_capacity // 2)


def eval_input(filename_queue):
    return tf.train.batch(read_file(filename_queue),
                          batch_size=qnd.FLAGS.batch_size,
                          capacity=qnd.FLAGS.batch_queue_capacity,
                          allow_smaller_final_batch=True)


def minimize(loss):
    return tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        0.01,
        "Adam")


def mnist_model(image, number):
    h = tf.contrib.layers.fully_connected(image, 200)
    h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(h, number))
    predictions = tf.argmax(h, axis=1)

    return predictions, loss, minimize(loss), {
        "accuracy": tf.reduce_mean(tf.to_float(tf.equal(predictions, number)))
    }


run = qnd.def_run()


def main():
    run(mnist_model,
        train_input,
        eval_input if qnd.FLAGS.use_eval_input_fn else None)


if __name__ == "__main__":
    main()
