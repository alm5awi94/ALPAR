"""
Python script for plotting and further evaluation based on the json data
output of the evaluation module and others
"""

# import json
import json
import os

import matplotlib.pyplot as plt

# ... use json to load eval data

layers = [3, 5, 8, 10, 13, 15, 17, 20, 22]
algos = ["dnnre", "alpa"]
time = {
    "alpa": [],
    "dnnre": []
}

for h in layers:
    for a in algos:
        with open(os.path.join(
                "rules", f"model_{h}h_{a}", "eval_metrics.json"), "r") as file:
            time[a].append(json.load(file)["time"])

# use pyplot to plot the data

for a in algos:
    plt.plot(layers, time[a], label=a, marker="o")
# make sure to produce meaningful graphs with a descriptive title and labels
plt.xlabel("number of hidden layers")
plt.ylabel("time [s]")
# plt.yscale("log")
plt.xticks(layers)
plt.legend()
plt.title("Extraction Time by Network Size")
# save the graph as pdf for later possible use with latex
plt.savefig("graphs/h_1a.pdf")
# also save a png for easy viewing
plt.savefig("graphs/h_1a.png")
