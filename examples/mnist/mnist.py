from gargparse import add_argument, ARGS
import qnd
import tensorflow as tf



add_argument("--batch_size", type=int, default=64)
add_argument("--batch_queue_capacity", type=int, default=1024)



def read_file(filename_queue):
  _, serialized = tf.TFRecordReader().read(filename_queue)

  int64_feature = tf.FixedLenFeature([], dtype=tf.int64)

  features = tf.parse_single_example(serialized, {
    "image_raw": tf.FixedLenFeature([], dtype=tf.string),
    "label": int64_feature,
  })

  image = tf.decode_raw(features["image_raw"], tf.uint8)
  image.set_shape([28**2])

  return tf.train.shuffle_batch(
      [tf.to_float(image) / 255 - 0.5, features["label"]],
      batch_size=ARGS.batch_size,
      capacity=ARGS.batch_queue_capacity,
      min_after_dequeue=ARGS.batch_queue_capacity//2)


def linear(h, num_outputs):
  return tf.contrib.layers.fully_connected(
      h,
      num_outputs=num_outputs)


def minimize(loss):
  return tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      0.01,
      "Adam")


def mnist_model(image, number):
  h = linear(image, num_outputs=42)
  h = linear(h, num_outputs=10)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(h, number))

  return tf.argmax(h, axis=1), loss, minimize(loss)


train_and_evaluate = qnd.def_train_and_evaluate()


def main():
  train_and_evaluate(mnist_model, read_file)



if __name__ == "__main__":
  main()
