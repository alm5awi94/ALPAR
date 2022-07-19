#!/bin/sh

# run all tests (works on linux only)
# current working directory must be tests

export PYTHONPATH=".."
python -m unittest discover
