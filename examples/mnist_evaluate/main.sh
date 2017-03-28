#!/bin/sh

. ../lib/mnist.sh &&


fetch_dataset &&

echo Training a MNIST model... &&
train &&

echo Evaluating a model with test data... &&
python3 evaluate.py \
  --infer_file $data_dir/test.tfrecords \
  --output_dir $var_dir/output
