#!/bin/sh

python -m rea "config/global.json" "config/mnist_conv_model.json" 2>&1 | tee "log_conv_data_model.txt"
python -m rea "config/global.json" "config/mnist_ff_model.json" 2>&1  | tee "log_ff_data_model.txt"
# model and data output are no part of repo due to size

# rule extraction and evaluation
echo "starting conv alpa"
python -m rea "config/global.json" "config/mnist_conv_alpa.json"   2>&1 | tee "log_conv_alpa.txt"
echo "starting ff alpa"
python -m rea "config/global.json" "config/mnist_ff_alpa.json" 2>&1   | tee "log_ff_alpa.txt"
echo "starting conv dnnre"
python -m rea "config/global.json" "config/mnist_conv_dnnre.json" 2>&1 | tee "log_conv_dnnre.txt"
echo "starting ff dnnre"
# dies after 2 hours due to increasing RAM usage (memkill)
python -m rea "config/global.json" "config/mnist_ff_dnnre.json" 2>&1 | tee "log_ff_dnnre.txt"
