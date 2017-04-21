#!/bin/sh

. ../lib/mnist.sh &&


git clean -dfx &&

fetch_dataset &&

python3 mnist_full.py $train_options
