#!/bin/sh

var_dir=var
data_dir=$var_dir/data
script=train.py


mnist() {
  python3 $script \
    --train_steps 1000 \
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
  ../lib/fetch_dataset.sh &&

  kill_servers

  mnist ps > $var_dir/ps.log 2>&1 &
  mnist master &&

  kill_servers
}


main
