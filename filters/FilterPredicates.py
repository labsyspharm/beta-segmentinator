import abc
import numpy
import torch


class Predicate(abc.ABC):
    """
    Abstract class that acts as a wrapper for other predicates, just ensures they contain the apply method.
    Should return @code{True} if the selected object should remain in the list, @code{False} if it does NOT pass the filter and should be removed.
    """
    @abc.abstractmethod
    def apply(self, x: dict[str, torch.tensor]):
        pass


class FilterPredicateHandler:
    """
    This class handles the predicates and aggregates the results.
    """

    def __init__(self):
        self.filters = list()

    def add_filter(self, filter: Predicate):
        if not isinstance(filter, Predicate):
            raise AttributeError("Variable \"filter\" is expected to inherit from class Predicate, but is instead {}".format(type(filter)))

        self.filters.append(filter)

    def apply(self, x: dict[str, torch.tensor]) -> numpy.ndarray[bool]:
        res = [f.apply(x) for f in self.filters]

        return numpy.logical_and.reduce(res)
