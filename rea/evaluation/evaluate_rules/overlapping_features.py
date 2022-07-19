# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/evaluate_rules
# added type-hints
# added support for more than one rule per class
# actually changed so much that this is not even close to the original anymore

from collections import OrderedDict, Set

from rules.rule import Rule


def overlapping_features(rules: Set[Rule]) -> OrderedDict:
    """
    Return the number of overlapping (common) features considered in output
    class rulesets.

    This computes the overall intersection of used features
    (i.e. features used in *every* rule) and per class intersection (i.e.
    features used in *every* rule that concludes to a certain class) and union
    (i.e. set of features used in rules concluding to the same class) of
    features.

    :example:

    .. code-block:: text

        h_0,1 > 0 AND h_0,1 < 1 -> class 0
        h_0,1 > 1 AND h_0,2 < 1 -> class 1
        h_0,2 < 1 AND h_0,3 < 0 --> class 1

        overall intersection: []
        per class intersection:
            class 0: [h_0,1]
            class 1: [h_0,2]
        per class union:
            class 0: [h_0,1]
            class 1: [h_0,1; h_0,2; h_0,3]

    :param rules: The ruleset to analyse

    :return: #common_features, common_features, features used in each class

    """
    # list of sets of features used in each rule indexed by class names
    features_per_class = {}
    for rule in rules:
        rule_features = set()  # set of features used in this rule
        # if the cls is not already present in the dict, initialise its value
        class_name = rule.get_conclusion().name
        if class_name not in features_per_class:
            features_per_class[class_name] = list()
        # get features used by the terms present in the current rule
        for clause in rule.get_premise():
            for term in clause.get_terms():
                # add features to set
                rule_features.add(term.get_neuron())
        # add all features from this rule as a separate set for the current cls
        features_per_class[class_name].append(rule_features)
    # calculate the union of features used for each class, i.e. all the
    # features that are used somewhere to conclude this class
    feature_union_per_class = {
        cls: set.union(*features_per_class[cls])
        for cls in features_per_class
    }
    # calculate the intersection of features used for each class, i.e. all the
    # features used in *every* rule concluding to this class
    feature_intersection_per_class = {
        cls: set.intersection(*features_per_class[cls])
        for cls in features_per_class
    }
    # get the overall common features (intersection of the union per class),
    # i.e. the features used in the rules of every class
    feature_intersection = set.intersection(*feature_union_per_class.values())
    return OrderedDict(
        feature_intersec_len=len(feature_intersection),
        feature_intersec=feature_intersection,
        feature_union_per_class=feature_union_per_class,
        feature_intersection_per_class=feature_intersection_per_class
    )
