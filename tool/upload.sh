#!/bin/sh

test/test.sh &&
git clean -dfx &&
python3 setup.py sdist bdist_wheel &&
twine upload dist/*
