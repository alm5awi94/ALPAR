#!/bin/sh

# This script is used to make calls to the REA CLI
echo "halloWorld"
python -m  "config/global.json" 2>&1 | tee "log.txt"

# more complex example
#python -m rea "config/global.json" "config/data.json" 2>&1 | tee "log.txt"
# python -m rea "config/global.json" "config/extract_alpa.json" 2>&1 | tee "log.txt"
# python -m rea "config/global.json" "config/extract_dnnre.json" 2>&1 | tee "log.txt"
# etc ...

# eval.py will be executed by the global run script
