#!/bin/sh

var_dir=var
data_dir=$var_dir/data
script=mnist_simple.py


mnist() {
  python3 $script \
    --num_epochs 2 \
    --train_file $data_dir/train.tfrecords \
    --eval_file $data_dir/validation.tfrecords \
    --output_dir $var_dir/output \
    --master_host localhost:2049 \
    --ps_hosts localhost:4242 \
    --task_type $1
}


kill_servers() {
  kill $(ps x | grep $script | grep -v grep | awk '{print $1}')
}


main() {
  if [ ! -d $data_dir ]
  then
    curl -SL https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/convert_to_records.py |
    python3 - --directory $data_dir
  fi

  kill_servers

  mnist ps > $var_dir/ps.log 2>&1 &
  mnist master

  kill_servers
}


main
