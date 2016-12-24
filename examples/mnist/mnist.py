import logging

import qnd
import tensorflow as tf


logging.getLogger().setLevel(logging.INFO)

qnd.add_flag("use_eval_input_fn", action="store_true")
qnd.add_flag("use_model_fn_ops", action="store_true")


def read_file(filename_queue):
    _, serialized = tf.TFRecordReader().read(filename_queue)

    scalar_feature = lambda dtype: tf.FixedLenFeature([], dtype)

    features = tf.parse_single_example(serialized, {
        "image_raw": scalar_feature(tf.string),
        "label": scalar_feature(tf.int64),
    })

    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image.set_shape([28**2])
    return tf.to_float(image) / 255 - 0.5, features["label"]


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
    train_op = minimize(loss)
    eval_metrics = {
        "accuracy": tf.reduce_mean(tf.to_float(tf.equal(predictions, number)))
    }

    if qnd.FLAGS.use_model_fn_ops:
        return tf.contrib.learn.estimators.model_fn.ModelFnOps(
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics)

    return predictions, loss, train_op, eval_metrics

run = qnd.def_run()


def main():
    run(mnist_model,
        read_file,
        read_file if qnd.FLAGS.use_eval_input_fn else None)


if __name__ == "__main__":
    main()
