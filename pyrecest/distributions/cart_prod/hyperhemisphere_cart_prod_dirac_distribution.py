import copy
from collections.abc import Callable

from ..abstract_dirac_distribution import AbstractDiracDistribution


class HyperhemisphereCartProdDiracDistribution(AbstractDiracDistribution):
    def __init__(self, d, w, dim_hemisphere, n_hemispheres):
        """
        Initialize a Dirac distribution with given Dirac locations and weights.

        :param d: Dirac locations as a numpy array.
        :param w: Weights of Dirac locations as a numpy array. If not provided, defaults to uniform weights.
        """
        super().__init__(d, w)
        self.dim_hemisphere = dim_hemisphere
        self.n_hemispheres = n_hemispheres
        assert self.d.shape[-1] == (
            (1 + dim_hemisphere) * n_hemispheres
        ), "Dimension is not correct."
        self.dim = dim_hemisphere * n_hemispheres

    def apply_function_component_wise(
        self, f: Callable, f_supports_multiple: bool = True
    ):
        """
        Apply a function to the Dirac locations and return a new distribution.

        :param f: Function to apply.
        :returns: A new distribution with the function applied to the locations.
        """
        assert f_supports_multiple, "Function must support multiple inputs."
        dist = copy.deepcopy(self)
        for i in range(self.n_hemispheres):
            dist.d[
                i * self.dim_hemisphere : (i + 1) * self.dim_hemisphere  # noqa: E203
            ] = f(self.d[i])
        return dist
