#!/bin/sh

venv_dir=venv

(
  tool/clean.sh &&

  python3 -m venv $venv_dir &&
  . $venv_dir/bin/activate &&

  tool/install_tensorflow.sh &&
  python3 setup.py install &&

  cd examples/mnist &&
  make
)
