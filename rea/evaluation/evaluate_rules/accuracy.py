"""
Evaluate accuracy of rules
"""

# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/evaluate_rules
import numpy as np


# source code optimized with numpy for many labels

def accuracy(predicted_labels: np.array, true_labels: np.array):
    num_labels = len(predicted_labels)
    assert (num_labels == len(
        true_labels)), "Error: number of labels inconsistent !"
    acc = np.sum(predicted_labels == true_labels) / num_labels
    return acc
