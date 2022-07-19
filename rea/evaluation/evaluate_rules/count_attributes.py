from typing import Set

from rea.rules.rule import Rule


def count_attr_per_class(rules: Set[Rule], weight_by_conf: bool = False) -> \
        dict[int, dict[int, float]]:
    """
    Creates a dictionary with classes as keys and attribute counts as values.
    The class and attribute keys are encodings, i.e. integers.

    :param rules: A set of rules in DNF
    :param weight_by_conf: Weight the attribute counts by confidence of rules

    :return: A dictionary with dictionaries per class and count per attribute

    """
    class_cts: dict[int, dict[int, float]] = dict()
    for rule in rules:
        output_class = rule.get_conclusion().name
        if output_class not in class_cts:
            class_cts[output_class] = dict()
        _count_attr_in_rule(class_cts[output_class], rule, weight_by_conf)
    return class_cts


def count_attributes(rules: Set[Rule], weight_by_conf: bool = False) -> \
        dict[int, float]:
    """
    Creates a dictionary with attributes as keys and counts as values.
    The keys are encodings, i.e. integers.
    :param rules: A set of rules in DNF
    :param weight_by_conf: Weight the attribute counts by confidence of rules
    :return: A dictionary with count per attribute
    """
    ct: dict[int, float] = dict()
    for rule in rules:
        _count_attr_in_rule(ct, rule, weight_by_conf)
    return ct


def _count_attr_in_rule(ct: dict[int, float], rule: Rule,
                        weight_by_conf: bool):
    for clause_set in rule.get_premise():
        conf = clause_set.get_confidence()
        for term in clause_set.get_terms():
            # threshold = term.get_threshold()
            attr = term.get_neuron_index()
            if attr not in ct:
                ct[attr] = 0.0
            if weight_by_conf:
                ct[attr] += conf
            else:
                ct[attr] += 1.0
