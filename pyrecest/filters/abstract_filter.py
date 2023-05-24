""" Abstract base class for all filters """
from abc import ABC, abstractmethod


class AbstractFilter(ABC):
    """Abstract base class for all filters."""
    def __init__(self, filter_state=None):
        self._filter_state = filter_state

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        assert isinstance(
            new_state, type(self._filter_state)
        ), "New distribution has to be of the same class as (or inherit from) the previous density."
        self._filter_state = new_state

    @abstractmethod
    def get_point_estimate(self):
        """Get the point estimate of the filter."""

    @property
    def dim(self):
        """Convenience function to get the dimension of the filter.
        Overwrite if the filter is not directly based on a distribution."""
        return self.filter_state.dim

    def plot_filter_state(self):
        """Plot the filter state."""
        self.filter_state.plot()
