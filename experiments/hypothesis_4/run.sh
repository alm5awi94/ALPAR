#!/bin/sh

# This script is used to make calls to the REA CLI

python -m rea "config/global.json" "config/wisconsin_ff_model.json" 2>&1  | tee "logs/log_data_model.txt"
python -m rea "config/global.json" "config/wisconsin_ff_alpa.json" 2>&1   | tee "logs/log_alpa.txt"
python -m rea "config/global.json" "config/wisconsin_ff_dnnre.json" 2>&1 | tee "logs/log_dnnre.txt"

# eval.py will be executed by the global run script
