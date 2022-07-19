#!/bin/sh

# Run this file to execute an experiment stored in a subfolder of this one.
# For this, the configuration name must equal the folder name.

# check empty argument
[ -z "$1" ] && { echo "Please specify path as first argument" ; exit 1; }
# step into the experiments dir
cd "$1" || { echo "Cannot find experiment $1" ; exit 1; }
# execute the experiment that is given as first argument
PYTHONPATH="../../" ./run.sh
PYTHONPATH="../.." python ./eval.py
# step back out of the dir
cd ..
