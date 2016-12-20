#!/bin/sh

for module in $(find qnd | grep '_test\.py' \
                         | sed 's/\(qnd\/.*_test\)\.py/\1/g' \
                         | tr / .)
do
  echo Testing $module
  python3 -m $module
done
