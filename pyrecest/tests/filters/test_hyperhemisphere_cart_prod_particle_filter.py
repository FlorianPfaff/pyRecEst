import unittest

import pyrecest.backend  # pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (  # pylint: disable=no-name-in-module,no-member,redefined-builtin
    abs,
    array,
    sum,
)
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.filters.hyperhemisphere_cart_prod_particle_filter import (
    HyperhemisphereCartProdParticleFilter,
)


class HyperHemisphereCartProdParticleFilterTest(unittest.TestCase):
    def test_init(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        self.assertEqual(
            pf.filter_state.d.shape,
            (n_particles, (dim_hemisphere + 1) * n_hemispheres),
        )

    @unittest.skipIf(
        pyrecest.backend.__name__
        in (
            "pyrecest.jax",
            "pyrecest.pytorch",
        ),  # pylint: disable=no-name-in-module,no-member
        reason="Backend not supported'",
    )
    def test_set_state(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        self.assertEqual(
            pf.filter_state.d.shape,
            (n_particles, (dim_hemisphere + 1) * n_hemispheres),
        )
        pf.filter_state = VonMisesFisherDistribution(array([0.0, 0.0, 0.0, 1.0]), 1.0)

    @unittest.skipIf(
        pyrecest.backend.__name__
        in (
            "pyrecest.jax",
            "pyrecest.pytorch",
        ),  # pylint: disable=no-name-in-module,no-member
        reason="Backend not supported'",
    )
    def test_predict(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        pf.filter_state = VonMisesFisherDistribution(array([0.0, 0.0, 0.0, 1.0]), 1.0)

        noise_distribution = VonMisesFisherDistribution(
            array([0.0, 0.0, 0.0, 1.0]), 1.0
        )

        def identity_function(x):
            return x

        pf.predict_nonlinear_each_part(identity_function, noise_distribution)

    @unittest.skipIf(
        pyrecest.backend.__name__
        in (
            "pyrecest.jax",
            "pyrecest.pytorch",
        ),  # pylint: disable=no-name-in-module,no-member
        reason="Backend not supported'",
    )
    def test_update(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        pf.filter_state = VonMisesFisherDistribution(array([0.0, 0.0, 0.0, 1.0]), 1.0)

        def likelihood_function(x):
            return abs(sum(x, axis=1))  # noqa: A001

        pf.update_nonlinear_using_likelihood(likelihood_function)


if __name__ == "__main__":
    unittest.main()
