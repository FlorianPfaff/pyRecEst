import importlib

# Import the classes in the current directory
from pyrecest.distributions.abstract_custom_distribution import AbstractCustomDistribution
from pyrecest.distributions.abstract_dirac_distribution import AbstractDiracDistribution
from pyrecest.distributions.abstract_disk_distribution import AbstractDiskDistribution
from pyrecest.distributions.abstract_ellipsoidal_ball_distribution import AbstractEllipsoidalBallDistribution
from pyrecest.distributions.abstract_periodic_distribution import AbstractPeriodicDistribution
from pyrecest.distributions.abstract_uniform_distribution import AbstractUniformDistribution

# Import the classes in the circle subdirectory
circle = importlib.import_module("pyrecest.distributions.circle")
globals().update(circle.__dict__)

# Import the classes in the hypersphere_subset subdirectory
hypersphere_subset = importlib.import_module("pyrecest.distributions.hypersphere_subset")
globals().update(hypersphere_subset.__dict__)

# Import the classes in the hypertorus subdirectory
hypertorus = importlib.import_module("pyrecest.distributions.hypertorus")
globals().update(hypertorus.__dict__)

# Import the classes in the nonperiodic subdirectory
nonperiodic = importlib.import_module("pyrecest.distributions.nonperiodic")
globals().update(nonperiodic.__dict__)


from .hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from .circle.wrapped_normal_distribution import WrappedNormalDistribution
from .hypertorus.hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution
from .circle.circular_dirac_distribution import CircularDiracDistribution
from .hypertorus.toroidal_dirac_distribution import ToroidalDiracDistribution
from .circle.von_mises_distribution import VonMisesDistribution
from .hypersphere_subset.von_mises_fisher_distribution import VonMisesFisherDistribution

# Aliases for brevity and compatibility with libDirectional
HypertoroidalWNDistribution = HypertoroidalWrappedNormalDistribution
WNDistribution = WrappedNormalDistribution
HypertoroidalWDDistribution = HypertoroidalDiracDistribution
ToroidalWDDistribution = ToroidalDiracDistribution
VMDistribution = VonMisesDistribution
WDDistribution = CircularDiracDistribution
VMFDistribution = VonMisesFisherDistribution

__all__ = [
    "AbstractCustomDistribution",
    "AbstractDiracDistribution",
    "AbstractDiskDistribution",
    "AbstractEllipsoidalBallDistribution",
    "AbstractPeriodicDistribution",
    "AbstractUniformDistribution",
    "HypertoroidalWNDistribution",
    "WNDistribution",
    "HypertoroidalWDDistribution",
    "ToroidalWDDistribution",
    "VMDistribution",
    "WDDistribution",
    "VMFDistribution"
] + circle.__all__ + hypersphere_subset.__all__ + hypertorus.__all__ + nonperiodic.__all__