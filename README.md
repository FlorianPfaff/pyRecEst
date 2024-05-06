# pyRecEst

*Recursive Bayesian Estimation for Python*

pyRecEst is a Python library tailored for recursive Bayesian estimation, compatible with numpy, pytorch, and jax backends.

Features of pyRecEst include:

* Distribution and Densities: Provides tools for handling distributions and densities across Euclidean spaces and manifolds.
* Filters and Trackers: Offers a suite of recursive Bayesian estimators (filters or trackers) for both Euclidean spaces and manifolds. This includes capabilities for:
** Multi-Target Tracking (MTT)
** Extended Object Tracking (EOT)
* Evaluation Framework: Contains an evaluation framework to facilitate comparison between different filters.
* Sampling Methods: Includes methods for sampling of the distributions and generating grids.

## Usage

Please refer to the test cases for usage examples.

## Credits

- Florian Pfaff (<pfaff@kit.edu>)

pyRecEst borrows its structure from libDirectional and follows its code closely for many classes. libDirectional, a project to which I contributed extensively, is [available on GitHub](https://github.com/libDirectional). The backend implementations are based on those of [geomstats](https://github.com/geomstats/geomstats).

## License
`pyRecEst` is licensed under the MIT License.
