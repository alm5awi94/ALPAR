from typing import Set

from . import DELETE_REDUNDANT_TERMS_FLAG
from .logic_manipulator.delete_redundant_terms import remove_redundant_terms
from .term import Term


# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/rules

class ConjunctiveClause:
    """
    Represent conjunctive clause. All terms in clause are ANDed together.
     Immutable and Hashable.

    Each conjunctive clause of terms has its own confidence value
    """
    __slots__ = ['terms', 'confidence', 'rank_score']

    def __init__(self, terms: Set[Term] = None, confidence=1):
        if terms is None:
            terms = set()

        rank_score = 0

        if DELETE_REDUNDANT_TERMS_FLAG:
            terms = remove_redundant_terms(terms)

        super(ConjunctiveClause, self).__setattr__('terms', terms)
        super(ConjunctiveClause, self).__setattr__('confidence', confidence)
        super(ConjunctiveClause, self).__setattr__('rank_score', rank_score)

    def __str__(self):
        terms_str = [str(term) for term in self.terms]
        return str(self.confidence) + '[' + ' AND '.join(terms_str) + ']'

    # def __setattr__(self, name, value):
    #     msg = "'%s' is immutable, can't modify %s" % (self.__class__, name)
    #     raise AttributeError(msg)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.terms == other.terms
        )

    def __hash__(self):
        x = hash(1)
        for term in self.terms:
            x = x ^ hash(term)
        return x

    def get_terms(self) -> Set[Term]:
        return self.terms

    def get_confidence(self) -> float:
        return self.confidence

    def set_rank_score(self, score):
        self.rank_score = score

    def get_rank_score(self):
        return self.rank_score

    def union(self, other) -> 'ConjunctiveClause':
        # Return new conjunctive clause that has all terms from both
        terms = self.terms.union(other.get_terms())
        confidence = self.confidence * other.get_confidence()
        # todo change this? see when called? its not right

        return ConjunctiveClause(terms=terms, confidence=confidence)

    def evaluate(self, data) -> bool:
        """
        Evaluate clause with data Dict[Neuron, float]
        """
        for term in self.terms:
            if not term.apply(data[term.get_neuron()]):
                return False

        # All conditions in the clause are satisfied
        return True
