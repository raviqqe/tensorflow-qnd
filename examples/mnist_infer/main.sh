#!/bin/sh

. ../lib/mnist.sh &&


fetch_dataset &&

echo Training a MNIST model... &&
train &&

echo Infering labels of test data... &&
infer > $prediction_file &&

echo Calculating test accuracy... &&
python3 gt.py $data_dir/test.tfrecords > $gt_file &&
python3 accuracy.py $prediction_file $gt_file
