"""
Evaluate fidelity of rules generated
 i.e. how well do they mimic the performance of the Neural Network
"""

# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/evaluate_rules
import numpy as np


# source code optimized with numpy for many labels

def fidelity(predicted_labels: np.ndarray, network_labels: np.ndarray):
    num_labels = len(predicted_labels)
    assert (num_labels == len(
        network_labels)), "Error: number of labels inconsistent !"

    fid = np.sum(predicted_labels == network_labels) / num_labels
    return fid
