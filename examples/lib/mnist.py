import qnd
import tensorflow as tf


def _preprocess_image(image):
    return tf.to_float(image) / 255 - 0.5


def read_file(filename_queue):
    _, serialized = tf.TFRecordReader().read(filename_queue)

    def scalar_feature(dtype): return tf.FixedLenFeature([], dtype)

    features = tf.parse_single_example(serialized, {
        "image_raw": scalar_feature(tf.string),
        "label": scalar_feature(tf.int64),
    })

    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image.set_shape([28**2])

    return _preprocess_image(image), features["label"]


def serving_input_fn():
    features = {
        'image': _preprocess_image(tf.placeholder(tf.uint8, [None, 28**2])),
    }

    return tf.contrib.learn.InputFnOps(features, None, features)


def minimize(loss):
    return tf.train.AdamOptimizer().minimize(
        loss,
        tf.contrib.framework.get_global_step())


def def_model():
    qnd.add_flag("hidden_layer_size", type=int, default=64,
                 help="Hidden layer size")

    def model(image, number=None, mode=None):
        h = tf.contrib.layers.fully_connected(image,
                                              qnd.FLAGS.hidden_layer_size)
        h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)

        predictions = tf.argmax(h, axis=1)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=number,
                                                           logits=h))

        return predictions, loss, minimize(loss), {
            "accuracy": tf.contrib.metrics.streaming_accuracy(predictions,
                                                              number)[1],
        }

    return model
