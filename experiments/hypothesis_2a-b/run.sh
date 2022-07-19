#!/bin/sh

# This script is used to make calls to the REA CLI

python -m rea "config/global.json" "config/data_model.json" 2>&1 | tee "log.txt"
python -m rea "config/global.json" "config/ex_eval_alpa.json" 2>&1 | tee "log.txt"
python -m rea "config/global.json" "config/ex_eval_dnnre.json" 2>&1 | tee "log.txt"

# eval.py will be executed by the global run script
