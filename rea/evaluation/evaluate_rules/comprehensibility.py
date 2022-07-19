"""
Compute comprehensibility of ruleset generated

- Number of rules per class = number of conjunctive clauses in a classes DNF
- Number of terms per rule: Min, Max, Average
"""
from collections import OrderedDict
from typing import Set, Tuple
from rules.rule import Rule


# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/evaluate_rules
# changed to accept multiple rules per class and produce accurate statistics
# added type-hints


def comprehensibility(rules: Set[Rule]) -> OrderedDict[Tuple]:
    # we use a dict because class_name might be a string -> no valid list index
    all_ruleset_info = {}

    for class_ruleset in rules:
        class_name = str(class_ruleset.get_conclusion().name)
        # if class not already in dict, add it
        if class_name not in all_ruleset_info:
            all_ruleset_info[class_name] = {
                "n_clauses_in_class": 0,  # number of rules for this class
                "min_n_terms": float("inf"),  # min number of terms in a rule
                "max_n_terms": 0,  # max number of terms in a rule of this cls
                "av_n_terms_per_class": 0  # now: total #terms, later average
            }
        # Number of rules in that class
        n_clauses_in_class = len(class_ruleset.get_premise())
        all_ruleset_info[class_name][
            "n_clauses_in_class"] += n_clauses_in_class
        #  Get min max average number of terms in a clause
        min_n_terms = float('inf')
        max_n_terms = 0
        total_n_terms = 0
        for clause in class_ruleset.get_premise():
            # Number of terms in the clause
            n_clause_terms = len(clause.get_terms())
            if n_clause_terms < min_n_terms:
                min_n_terms = n_clause_terms
            if n_clause_terms > max_n_terms:
                max_n_terms = n_clause_terms
            total_n_terms += n_clause_terms
        # update entry for class_name in dict
        all_ruleset_info[class_name]["min_n_terms"] = min(
            all_ruleset_info[class_name]["min_n_terms"], min_n_terms)
        all_ruleset_info[class_name]["max_n_terms"] = max(
            all_ruleset_info[class_name]["max_n_terms"], max_n_terms)
        # this field contains the total number for now, but is updated later
        all_ruleset_info[class_name][
            "av_n_terms_per_class"] += total_n_terms

    # update the av_n_terms_per_rule field to contain the average number of
    # terms in a rule for the specific class
    for vls in all_ruleset_info.values():
        if vls["n_clauses_in_class"] > 0:
            vls["av_n_terms_per_class"] = round(
                vls["av_n_terms_per_class"] / vls["n_clauses_in_class"], 2
            )
        else:
            vls["av_n_terms_per_rule"] = 0
    output_classes = list(all_ruleset_info.keys())
    # transform dict indexed by class to list indexed by statistic, e.g.
    # dict(cls: vals) to list of shape [vals, cls]
    values = list(
        map(lambda x: [y for y in x.values()], all_ruleset_info.values()))
    n_clauses_per_class, min_n_terms, max_n_terms, av_n_terms_per_rule = zip(
        *values)
    n_rules = len(rules)
    return OrderedDict(output_classes=output_classes,
                       n_rules=n_rules,
                       n_clauses_per_class=n_clauses_per_class,
                       min_n_terms=min_n_terms,
                       max_n_terms=max_n_terms,
                       av_n_terms_per_rule=av_n_terms_per_rule)
