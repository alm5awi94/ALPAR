"""
Python script for plotting and further evaluation based on the json data
output of the evaluation module and others
"""

# import json
import matplotlib.pyplot as plt

# ... use json to load eval data

# use pyplot to plot the data

plt.plot([1, 2, 3], [1, 2, 3], label="linear curve")
# make sure to produce meaningful graphs with a descriptive title and labels
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# save the graph as pdf for later possible use with latex
plt.savefig("graphs/example.pdf")
# also save a png for easy viewing
plt.savefig("graphs/example.png")
