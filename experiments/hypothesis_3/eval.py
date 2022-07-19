"""
Python script for plotting and further evaluation based on the json data
output of the evaluation module and others
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rea.data.data import Data
from rea.evaluation.evaluate_rules.count_attributes import \
    count_attr_per_class, \
    count_attributes
from rea.evaluation.evaluation import Evaluation

data = Data.read("data/ff")


def plot_heatmap(ct_dict, ax: plt.axis,
                 vmax=None) -> matplotlib.image.AxesImage:
    heatmap = np.zeros((28, 28))
    for attr, ct in ct_dict.items():
        feature = data.feature_names[attr].split('x')
        x, y = int(feature[0]) - 1, int(feature[1]) - 1
        heatmap[x, y] = ct
    return ax.imshow(heatmap, vmin=0, vmax=vmax)


# Load extracted rules and count attributes in premises
for rule_path, fig_name in (
        ("alpa_rules/conv", "alpa_conv"), ("alpa_rules/ff", "alpa_ff"),
        ("dnnre_rules/ff", "dnnre_ff"), ("dnnre_rules/conv", "dnnre_conv")):
    print(f"eval.py for {rule_path} ({fig_name})")
    try:
        rules = Evaluation.read_rules(rule_path)
    except FileNotFoundError as e:
        print("No rules were created ", e)
        continue
    # count occurrence of attribute over all rules:
    ct_total = count_attributes(rules)
    # count occurrence of attribute per class:
    ct_per_cls = count_attr_per_class(rules)
    print(
        f"Found {len(rules)} rules. Out of {data.num_features} attributes "
        f"{len(ct_total)} attributes are used for classification.")
    # print attribute occurrences per class
    for output_class in data.class_names:
        if output_class in ct_per_cls:
            num_attrs = len(ct_per_cls[output_class])
        else:
            num_attrs = 0
        print(
            f"class {output_class} predicted by {num_attrs} attributes")

    # for all classes and class 7
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    ax[0].set_title("All classes")
    ax[1].set_title("Class 7")
    im = plot_heatmap(ct_per_cls[7], ax=ax[1])
    f.colorbar(im, ax=ax[1], location="right", fraction=0.1)
    im = plot_heatmap(ct_total, ax=ax[0])
    f.colorbar(im, ax=ax[0], location="left", fraction=0.1)
    f.savefig(f"graphs/h_2_all_{fig_name}.pdf")
    f.savefig(f"graphs/h_2_all_{fig_name}.png")
    f, axis = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(25, 8))
    axis = axis.flatten()
    vmax = max(max(v.values()) for v in ct_per_cls.values())
    for ax, output_class in zip(axis, data.class_names):
        ax.set_title(str(output_class))
        if output_class in ct_per_cls:
            im = plot_heatmap(ct_per_cls[output_class], ax=ax, vmax=vmax)
    f.colorbar(im, ax=axis.ravel().tolist(), fraction=0.1)
    f.savefig(f"graphs/h_2_cls_{fig_name}.pdf")
    f.savefig(f"graphs/h_2_cls_{fig_name}.png")
