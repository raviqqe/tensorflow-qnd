#!/bin/sh

var_dir=var
data_dir=$var_dir/data
gt_file=$var_dir/gt.csv
prediction_file=$var_dir/predictions.csv
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
    --output_dir $var_dir/output
}


kill_servers() {
  kill $(ps x | grep $train_script | grep -v grep | awk '{print $1}')
}


main() {
  ../lib/fetch_dataset.sh &&

  kill_servers

  echo Training a MNIST model...

  train ps > $var_dir/train_ps.log 2>&1 &
  train master &&

  kill_servers || exit 1

  echo Infering labels of test data...

  infer > $prediction_file &&
  python3 gt.py $data_dir/test.tfrecords > $gt_file &&

  python3 accuracy.py $prediction_file $gt_file
}


main
