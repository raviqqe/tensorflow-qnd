#!/bin/sh

var_dir=var
data_dir=$var_dir/data
shared_data_dir=../$data_dir


main() {
  if [ ! -d $shared_data_dir ]
  then
    curl -SL https://raw.githubusercontent.com/raviqqe/tensorflow/patch-1/tensorflow/examples/how_tos/reading_data/convert_to_records.py |
    python3 - --directory $shared_data_dir
  fi &&

  if [ ! -d $data_dir ]
  then
    mkdir -p $(dirname $data_dir) &&
    ln -s ../$shared_data_dir $data_dir
  fi
}


main
