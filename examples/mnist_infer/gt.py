import sys

import tensorflow as tf


def main():
    for serialized in tf.python_io.tf_record_iterator(sys.argv[1]):
        example = tf.train.Example()
        example.ParseFromString(serialized)
        print(*example.features.feature["label"].int64_list.value)


if __name__ == "__main__":
    main()
