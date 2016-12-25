#!/bin/sh

var_dir=var
data_dir=$var_dir/data
script=mnist_multi_workers.py
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
  if [ ! -d $data_dir ]
  then
    curl -SL https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/convert_to_records.py |
    python3 - --directory $data_dir
  fi

  kill_servers

  mnist ps > $var_dir/ps.log 2>&1 &

  worker_id=0
  for worker in echo $(echo $workers | tr , ' ')
  do
    mnist worker --task_index $worker_id > $var_dir/worker-$worker_id.log 2>&1 &
    worker_id=$(expr $worker_id + 1)
  done

  mnist master &&

  kill_servers
}


main
