import tensorflow as tf

import mnist


def test_read_file():
    data = mnist.read_file(tf.train.string_input_producer(
        tf.matching_files('examples/var/data/train.tfrecords')))

    with tf.Session() as session:
        tf.train.queue_runner.start_queue_runners(session)
        print(session.run(data))
