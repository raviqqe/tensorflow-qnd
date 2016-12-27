#!/bin/sh

var_dir=var
data_dir=$var_dir/data
train_script=train.py
infer_script=infer.py


train() {
  python3 $train_script \
    --train_steps 1000 \
    --train_file $data_dir/train.tfrecords \
    --eval_file $data_dir/validation.tfrecords \
    --output_dir $var_dir/output \
    --master_host localhost:2049 \
    --ps_hosts localhost:4242 \
    --task_type $1
}

infer() {
  python3 $infer_script \
    --infer_file $data_dir/test.tfrecords \
    --output_dir $var_dir/output \
    --master_host localhost:2049 \
    --ps_hosts localhost:4242 \
    --task_type $1
}


kill_servers() {
  kill $(ps x | grep -e $train_script -e $infer_script | grep -v grep |
         awk '{print $1}')
}


main() {
  if [ ! -d $data_dir ]
  then
    curl -SL https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/convert_to_records.py |
    python3 - --directory $data_dir
  fi

  kill_servers

  echo Training a MNIST model...

  train ps > $var_dir/train_ps.log 2>&1 &
  train master &&

  kill_servers

  echo Infering labels of test dataset...

  infer ps > $var_dir/infer_ps.log 2>&1 &
  infer master > $var_dir/predictions.csv
}


main
