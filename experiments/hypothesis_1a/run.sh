#!/bin/sh

# This script is used to make calls to the REA CLI

# generate configs
cd config || exit
python gen.py
cd ..

# pre-process data
python -m rea "config/global.json" "config/data.json"

# run configs
for m in config/model_*; do
  echo "training model $m" 2>&1 | tee "log.txt"
  python -m rea "config/global.json" "$m" 2>&1 | tee "log.txt"
done

for e in config/ex_*; do
  echo "extraction using $e" 2>&1 | tee "log.txt"
  python -m rea "config/global.json" "$e" 2>&1 | tee "log.txt"
done

# eval.py will be executed by the global run script
