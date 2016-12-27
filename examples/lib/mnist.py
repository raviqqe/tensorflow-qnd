import tensorflow as tf


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
        0.001,
        "Adam")


def model(image, number=None, mode=None):
    h = tf.contrib.layers.fully_connected(image, 64)
    h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)

    predictions = tf.argmax(h, axis=1)

    if mode == tf.contrib.learn.ModeKeys.INFER:
        return predictions

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(h, number))

    return predictions, loss, minimize(loss), {
        "accuracy": tf.contrib.metrics.streaming_accuracy(predictions,
                                                          number)[1],
    }
