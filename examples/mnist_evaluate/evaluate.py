import logging

import qnd

import mnist


evaluate = qnd.def_evaluate()


model = mnist.def_model()


def main():
    logging.getLogger().setLevel(logging.INFO)
    print(evaluate(model, mnist.read_file))


if __name__ == "__main__":
    main()
