from typing import Set

from .helpers import terms_set_to_neuron_dict
from ..clause import ConjunctiveClause
from ..term import TermOperator


# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/logic_manipulator

def is_satisfiable(clause: ConjunctiveClause):
    """
    Return whether or not the clause is satisfiable. Unsatisfiable if empty
     or a neurons min value >= its max value
    """
    # Empty Clause
    if len(clause.get_terms()) == 0:
        return False

    # Check if neurons min value >= max value
    neuron_conditions = terms_set_to_neuron_dict(clause.get_terms())
    for neuron in neuron_conditions.keys():
        # If neuron is specified with <= and >
        if neuron_conditions[neuron][TermOperator.GreaterThan] and \
                neuron_conditions[neuron][TermOperator.LessThanEq]:
            gt_vals = neuron_conditions[neuron][TermOperator.GreaterThan]
            lteq_vals = neuron_conditions[neuron][TermOperator.LessThanEq]

            if gt_vals and lteq_vals:
                # if neuron is subject to both predicates
                min_value = min(gt_vals)
                max_value = max(lteq_vals)
                if min_value >= max_value:
                    return False

    # All conditions on a neuron are satisfiable
    return True


def remove_unsatisfiable_clauses(clauses: Set[ConjunctiveClause]):
    """
    Remove unsatisfiable clauses
    """
    satisfiable_clauses = set()
    for clause in clauses:
        if is_satisfiable(clause):
            satisfiable_clauses.add(clause)

    return satisfiable_clauses
