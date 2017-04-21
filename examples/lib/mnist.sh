var_dir=var
data_dir=$var_dir/data
shared_data_dir=../$data_dir
gt_file=$var_dir/gt.csv
prediction_file=$var_dir/predictions.csv

train_options=\
'--train_steps 1000 '\
'--eval_steps 50 '\
"--train_file $data_dir/train.tfrecords "\
"--eval_file $data_dir/validation.tfrecords "\
"--output_dir $var_dir/output"


train() {
  python3 train.py $train_options
}


infer() {
  python3 infer.py \
    --infer_file $data_dir/test.tfrecords \
    --output_dir $var_dir/output
}


fetch_dataset() {
  if [ ! -d $shared_data_dir ]
  then
    curl -SL https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py |
    python3 - --directory $shared_data_dir
  fi &&

  if [ ! -d $data_dir ]
  then
    mkdir -p $(dirname $data_dir) &&
    ln -s ../$shared_data_dir $data_dir
  fi
}
