#!/bin/sh

var_dir=var
data_dir=$var_dir/data


main() {
  ../lib/fetch_dataset.sh &&
  python3 train.py \
    --train_steps 1000 \
    --train_file $data_dir/train.tfrecords \
    --eval_file $data_dir/validation.tfrecords \
    --output_dir $var_dir/output
}


main
