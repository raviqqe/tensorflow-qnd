#!/bin/sh

. ../lib/mnist.sh &&


git clean -dfx &&

fetch_dataset &&

python3 mnist_full.py \
  --train_steps 1000 \
  --train_file $data_dir/train.tfrecords \
  --eval_file $data_dir/validation.tfrecords \
  --output_dir $var_dir/output
