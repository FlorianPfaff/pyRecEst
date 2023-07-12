import importlib
import pkgutil

# Iterate over each module in the current package
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # If it's a package, skip it
    if is_pkg:
        continue

    # Form the full module name (e.g., 'pyrecest.distributions.module_name')
    full_module_name = f"{__name__}.{module_name}"

    # Import the module and add it to globals
    module = importlib.import_module(full_module_name)
    globals().update({n: getattr(module, n) for n in module.__all__})

# Now __all__ includes all symbols that any of the modules define in their __all__
__all__ = list(globals().keys())
# Aliases for brevity and compatibility with libDirectional
# HypertoroidalWNDistribution = HypertoroidalWrappedNormalDistribution
# WNDistribution = WrappedNormalDistribution
# HypertoroidalWDDistribution = HypertoroidalDiracDistribution
# ToroidalWDDistribution = ToroidalDiracDistribution
# VMDistribution = VonMisesDistribution
# WDDistribution = CircularDiracDistribution
# VMFDistribution = VonMisesFisherDistribution

__all__ = [
    "CustomCircularDistribution",
    "HypertoroidalWrappedNormalDistribution",
    "VMDistribution",
    "HypertoroidalWDDistribution",
    "ToroidalWDDistribution",
    "VMFDistribution",
    "WDDistribution",
    "WNDistribution",
    "AbstractHemisphericalDistribution",
    "CustomLinearDistribution",
    "HyperhemisphericalWatsonDistribution",
    "AbstractLinearDistribution",
    "AbstractCircularDistribution",
    "AbstractDiracDistribution",
    "AbstractDiskDistribution",
    "AbstractEllipsoidalBallDistribution",
    "AbstractHyperhemisphericalDistribution",
    "AbstractHypersphereSubsetDistribution",
    "AbstractHypersphereSubsetUniformDistribution",
    "AbstractHypersphericalDistribution",
    "AbstractHypertoroidalDistribution",
    "AbstractPeriodicDistribution",
    "AbstractToroidalDistribution",
    "AbstractUniformDistribution",
    "BinghamDistribution",
    "AbstractCustomDistribution",
    "CustomHemisphericalDistribution",
    "CustomHyperhemisphericalDistribution",
    "DiskUniformDistribution",
    "EllipsoidalBallUniformDistribution",
    "CircularFourierDistribution",
    "GaussianDistribution",
    "HyperhemisphericalUniformDistribution",
    "HypersphericalMixture",
    "HypersphericalUniformDistribution",
    "HypertoroidalWDDistribution",
    "HypertoroidalWNDistribution",
    "ToroidalWDDistribution",
    "VonMisesFisherDistribution",
    "VonMisesDistribution",
    "WatsonDistribution",
    "CircularDiracDistribution",
    "WrappedNormalDistribution",
]
