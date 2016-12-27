import logging

import qnd

import mnist


train_and_evaluate = qnd.def_train_and_evaluate()


def main():
    logging.getLogger().setLevel(logging.INFO)
    train_and_evaluate(mnist.model, mnist.read_file)


if __name__ == "__main__":
    main()
