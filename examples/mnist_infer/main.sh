#!/bin/sh

var_dir=var
data_dir=$var_dir/data
gt_file=$var_dir/gt.csv
prediction_file=$var_dir/predictions.csv


train() {
  python3 train.py \
    --train_steps 1000 \
    --train_file $data_dir/train.tfrecords \
    --eval_file $data_dir/validation.tfrecords \
    --output_dir $var_dir/output
}


infer() {
  python3 infer.py \
    --infer_file $data_dir/test.tfrecords \
    --output_dir $var_dir/output
}


main() {
  ../lib/fetch_dataset.sh &&

  echo Training a MNIST model... &&
  train &&

  echo Infering labels of test data... &&
  infer > $prediction_file &&

  echo Calculating test accuracy... &&
  python3 gt.py $data_dir/test.tfrecords > $gt_file &&
  python3 accuracy.py $prediction_file $gt_file
}


main
