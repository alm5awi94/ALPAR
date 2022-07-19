"""
Helper methods
"""

import json
import re
from typing import Dict, Union

# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/rules
# TODO check W605 invalid escape sequence '\.'
import numpy as np

from data.data import Data
from rules.clause import ConjunctiveClause
from rules.rule import OutputClass, Rule
from rules.term import Neuron, Term

int_and_float_re = re.compile("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
bool_re = re.compile("((True)|(False))")


def str_to_bool(bool_str: str):
    if bool_str == 'True':
        return True
    elif bool_str == 'False':
        return False


def parse_variable_str_to_dict(variables_str) -> \
        Dict[str, Union[str, int, float, bool]]:
    """
    Parse string of variables of the form
        'variable_name="val" variable_name="val" variable_name="val"' into dict

    Where variable vals are cast to the correct type.
     This is the form C5 stores output data
    """
    variables = {}

    for var_str in variables_str.split(' '):
        if var_str != '':

            var_name = var_str.split('=')[0]
            var_value = var_str.split('=')[1].replace('"', '')

            # Cast to correct type
            # todo change this with just normal casting
            #  see if makes a difference timewise?
            if re.match(int_and_float_re, var_value):
                var_value = json.loads(
                    var_value)
            elif re.match(bool_re, var_value):
                var_value = str_to_bool(var_value)

            variables[var_name] = var_value

    return variables


def pretty_string_repr(ruleset: set[Rule], data: Data,
                       convert_classes: bool = False) -> list[str]:
    """
    Converts a set of rules into a list of human-readable rule strings.
    :param ruleset: The set to convert as strings.
    :param data: Instance to decode feature and class names.
    :param convert_classes: Whether classes should be converted or not.
    :return: A list of rule string representations.
    """
    # This is essentially just the out rolled code of the `__str__` methods
    # of Rule, Clause and Term instances.
    # Added to avoid changing the internal representation of the dnnre rules.
    ruleset_str_repr = []
    for rule in ruleset:
        premise: set[ConjunctiveClause] = rule.premise
        clause_strings = []
        for clause in premise:
            terms: set[Term] = clause.terms
            terms_strings = []
            for term in terms:
                attr_index = term.neuron.get_index()
                if data.scale_data:
                    threshold = data.inverse_transform_scaling(
                        term.threshold, attr_index)
                else:
                    threshold = term.threshold
                terms_strings.append(
                    '(' + str(data.feature_names[attr_index]) + ' ' + str(
                        term.operator) + ' ' + str(threshold) + ')')
            clause_strings.append(
                str(clause.confidence) + '[' + ' AND '.join(
                    terms_strings) + ']')

        if convert_classes and isinstance(rule.conclusion, OutputClass):
            conclusion = data.inverse_transform_classes(
                np.array([rule.conclusion.encoding]))[0]
        else:
            conclusion = rule.conclusion
        rule_str = "IF " + (' OR '.join(clause_strings)) + " THEN " + str(
            conclusion)
        ruleset_str_repr.append(rule_str)
    return ruleset_str_repr


def neuron_to_str(neuron: Neuron, data: Data) -> str:
    """
    Converts an input neuron name to a corresponding feature name.
    WARNING: does not work for hidden layer neurons (h_x_y where x > 0)!
    :param neuron: Neuron to get the corresponding feature name for
    :param data: Data instance which has the mapping for feature names
    :return: The name of the corresponding feature as a string
    """
    return data.feature_names[neuron.get_index()]
