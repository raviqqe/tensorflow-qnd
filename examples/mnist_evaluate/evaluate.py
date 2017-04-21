import logging

import qnd
import tensorflow as tf

import mnist


evaluate = qnd.def_evaluate()


model = mnist.def_model()


def main():
    logging.getLogger().setLevel(logging.INFO)

    try:
        print(evaluate(model, mnist.read_file))
    except tf.errors.OutOfRangeError:
        pass


if __name__ == "__main__":
    main()
