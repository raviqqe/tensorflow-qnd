import logging

import qnd

import mnist


infer = qnd.def_infer()


def read_file(filename_queue):
    return mnist.read_file(filename_queue)[0]


def main():
    logging.getLogger().setLevel(logging.INFO)

    for label in infer(mnist.model, read_file):
        print(label)


if __name__ == "__main__":
    main()
