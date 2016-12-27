#!/bin/sh

var_dir=var
data_dir=$var_dir/data
script=train.py
workers=localhost:19310,localhost:10019


mnist() {
  python3 $script \
    --train_steps 1000 \
    --train_file $data_dir/train.tfrecords \
    --eval_file $data_dir/validation.tfrecords \
    --output_dir $var_dir/output \
    --master_host localhost:2049 \
    --worker_hosts $workers \
    --ps_hosts localhost:4242 \
    --task_type "$@"
}


kill_servers() {
  kill $(ps x | grep $script | grep -v grep | awk '{print $1}')
}


main() {
  ../lib/fetch_dataset.sh &&

  kill_servers

  mnist ps > $var_dir/ps.log 2>&1 &

  worker_id=0
  for worker in $(echo $workers | tr , ' ')
  do
    mnist worker --task_index $worker_id > $var_dir/worker-$worker_id.log 2>&1 &
    worker_id=$(expr $worker_id + 1)
  done

  mnist master &&

  kill_servers
}


main
