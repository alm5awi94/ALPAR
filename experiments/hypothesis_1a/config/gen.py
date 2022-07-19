# This creates some configuration files for the experiment, because doing it
# manually would be tedious. Also we can try different parameters later.

import json
import os

hidden_layer_nums = [3, 5, 8, 10, 13, 15, 17, 20, 22]
algos = ["dnnre", "alpa"]
model_templ = {
    "data_path": "data/",
    "nwtype": "ff",
    "hidden_layer_units": [],
    "hidden_layer_activations": [],
    "use_class_weights": True,
    "batch_size": 1,
    "dropout": 0.2,
    "val_split": 0.25,
    "learning_rate": 0.001,
    "epochs": 0,
    "output_path": "model/"
}
ext_templ = {
    "data_path": "data/",
    "trained_model_path": "model/",
    "algorithm": "",
    "rules_dir": "rules/"
}

# generate model configs
for idx, h in enumerate(hidden_layer_nums):
    config_name = f"model_{h}h.json"
    # flipped parabola with center h // 2 and 0 at 0 and h
    # flattened to a maximum of 100 hidden units
    # increase size of hidden layers until center and then decrease
    hidden_layer_units = [
        min((-i * (i + 1 - h)) // 2 + 64, 80)
        for i in range(0, h)
    ]
    # hidden_layer_units = [64] * h
    # always use relu
    hidden_layer_activations = ["relu" for _ in range(h)]
    print(f"Creating config with {hidden_layer_units}")
    config_contents = model_templ.copy()
    config_contents["hidden_layer_units"] = hidden_layer_units
    config_contents["hidden_layer_activations"] = hidden_layer_activations
    # longer training for deeper networks
    config_contents["epochs"] = max(h * 25, 200)
    config_contents["learning_rate"] = 1.0 / h * 0.001
    config_contents["output_path"] = os.path.join(
        config_contents["output_path"], config_name[:-5])
    with open(config_name, "w+") as file:
        json.dump({"model": config_contents}, file, indent=4)

for idx, h in enumerate(hidden_layer_nums):
    model_name = f"model_{h}h"
    config_contents = ext_templ.copy()
    model_path = os.path.join(
        config_contents["trained_model_path"],
        model_name
    )
    config_contents["trained_model_path"] = model_path
    for algo in algos:
        config_name = f"ex_{algo}_{h}h.json"
        rules_path = os.path.join(
            "rules",
            f"{model_name}_{algo}"
        )
        print(f"writing config {config_name}")
        config_contents["algorithm"] = algo
        config_contents["rules_dir"] = rules_path
        with open(config_name, "w+") as file:
            json.dump({"extraction": config_contents}, file, indent=4)
