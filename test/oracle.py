import qnd
import tensorflow as tf


def model_fn(x, y):
    return (y,
            0.0,
            tf.contrib.framework.get_or_create_global_step().assign_add())


def input_fn(q):
    shape = (100,)
    return tf.zeros(shape, tf.float32), tf.ones(shape, tf.int32)


run = qnd.def_run()


def main():
    run(model_fn, input_fn)


if __name__ == "__main__":
    main()
