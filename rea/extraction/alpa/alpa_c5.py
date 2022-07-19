import logging
from typing import Set

import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
# Interface to R running embedded in a Python process
from rpy2.robjects.packages import importr

from rules.helpers import parse_variable_str_to_dict
from rules.rule import OutputClass, Rule
from rules.term import Neuron, Term

# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/rules
# adapted to work for ALPA rule generation

logger = logging.getLogger(__name__)

# activate Pandas conversion between R objects and Python objects
pandas2ri.activate()

# C50 R package is interface to C5.0 classification model
C50 = importr('C50')
C5_0 = robjects.r('C5.0')
C5_0_predict = robjects.r("predict.C5.0")


def _parse_C5_rule_str(rule_str: str) -> Set[Rule]:
    """
    Parse the string returned by the R C5 implementation into a set of rules.

    :param rule_str: The output of the C5 algorithm

    :return: A set of rules in DNF

    """
    rules_set: Set[Rule] = set()
    rule_str_lines = rule_str.split('\n')
    # skip the first to metadata lines
    line_index = 2
    # retrieve number of rules
    metadata_variables = parse_variable_str_to_dict(rule_str_lines[line_index])
    n_rules = metadata_variables['rules']
    # loop over all rules
    for _ in range(0, n_rules):
        line_index += 1
        rule_data_variables = parse_variable_str_to_dict(
            rule_str_lines[line_index])
        n_rule_terms = rule_data_variables['conds']
        # rule_conclusion = rule_conclusion_map[(rule_data_variables['class'])]
        rule_conclusion = OutputClass(name=rule_data_variables["class"],
                                      encoding=rule_data_variables["class"])
        # C5 rule confidence=
        # (number of training cases correctly classified + 1)
        # / (total training cases covered  + 2)
        rule_confidence = (rule_data_variables['ok'] + 1) / (
            rule_data_variables['cover'] + 2)
        # loop over all terms in the precondition
        rule_terms: Set[Term] = set()
        for _ in range(0, n_rule_terms):
            line_index += 1
            term_variables = parse_variable_str_to_dict(
                rule_str_lines[line_index])
            term_neuron_str = term_variables['att']
            # in ALPA, the neuron is always an input neuron (corresponding
            # to an attribute)
            term_neuron = Neuron(layer=0,
                                 index=term_neuron_str)
            # In C5, < -> <=, > -> >
            term_operator = '<=' if term_variables[
                'result'] == '<' else '>'
            term_operand = term_variables['cut']

            rule_terms.add(Term(neuron=term_neuron, operator=term_operator,
                                threshold=term_operand))
        rules_set.add(
            Rule.from_term_set(premise=rule_terms, conclusion=rule_conclusion,
                               confidence=rule_confidence))
    return rules_set


def get_c5_model(x: np.ndarray, y: np.ndarray,
                 seed: int = 42) -> robjects.vectors.ListVector:
    """Trains a C5.0 rule classifier with R on the training set x.

    :param x: Training data for the rule classifier
    :param y: Labels for the training data
    :param seed: Seed to use in the R implementation of C5.0

    :return: An instance from ``robjects.r('C5.0').C5_0`` call
    """
    # y has to be a factor vector for the R implementation
    y = robjects.vectors.FactorVector(list(map(str, pd.Series(y))))
    # Default = C5.0Control(subset = TRUE, bands = 0, winnow = FALSE,
    # noGlobalPruning = FALSE, CF = 0.25, minCases = 2,
    # fuzzyThreshold=FALSE, sample = 0, seed = sample.int(4096, size = 1) -1L,
    # earlyStopping = TRUE, label = "outcome")
    return C50.C5_0(x=pd.DataFrame(x), y=y, rules=True,
                    control=C50.C5_0Control(winnow=True, seed=seed,
                                            subset=True))


def c5_r_predict(c5_model: robjects.vectors.ListVector,
                 x: np.ndarray) -> np.ndarray:
    """
    Calls the R implementation to predict x with the R C5.0 Model.

    :param c5_model: instance from robjects.r('C5.0')
    :param x: Data to predict classes for

    :return: A numpy array with the indices of predicted classes

    """
    prediction_probs = C5_0_predict(c5_model, pd.DataFrame(x), type="prob")
    return np.argmax(prediction_probs, axis=1)


def get_c5_rules(c5_model: robjects.vectors.ListVector) -> Set[Rule]:
    """
    Parses rules from the c5_model instance to a python representation

    :param c5_model: instance from ``robjects.r('C5.0')``

    :return: A set of rules in DNF

    """
    C5_rules_str = c5_model.rx2('rules')[0]
    C5_rules = _parse_C5_rule_str(C5_rules_str)
    # logger.debug(C5_rules_str)
    return C5_rules
