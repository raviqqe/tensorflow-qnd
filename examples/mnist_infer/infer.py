import logging

import qnd

import mnist


infer = qnd.def_infer()


def main():
    logging.getLogger().setLevel(logging.INFO)

    for label in infer(mnist.model, mnist.read_file):
        print(label)


if __name__ == "__main__":
    main()
