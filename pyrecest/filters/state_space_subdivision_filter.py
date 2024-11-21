import numpy as np
import warnings
from scipy.stats import multivariate_normal
from .abstract_hypercylindrical_filter import AbstractHypercylindricalFilter
from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import SdHalfCondSdHalfGridDistribution

class StateSpaceSubdivisionFilter(AbstractHypercylindricalFilter):
    def __init__(self):
        self.apd = None

    def setState(self, apd_):
        if not self.apd.is_empty() and (len(self.apd.gaussians) != len(apd_.gaussians)):
            warnings.warn("Number of components differ.", "LinPeriodic:NoComponentsDiffer")
        self.apd = apd_

    def predictLinear(self, transitionDensity=None, covarianceMatrices=None, systemMatrices=None, linearInputVectors=None):
        if transitionDensity is not None:
            # Various assert checks
            pass
        n_areas = len(self.apd.linearDistributions)

        if transitionDensity is None and covarianceMatrices is None and systemMatrices is None and linearInputVectors is None:
            # Case 1
            pass
        elif transitionDensity is None:
            # Case 2
            pass
        else:
            # Case 3
            pass

    def update(self, likelihoodPeriodicGrid=None, likelihoodsLinear=None):
        if isinstance(likelihoodPeriodicGrid, AbstractDistribution):
            likelihoodPeriodicGrid = likelihoodPeriodicGrid.pdf(self.apd.gd.getGrid()).T
            
        if likelihoodPeriodicGrid is None and likelihoodsLinear is None:
            warnings.warn("Nothing to do for this update step.", "StateSpaceSubdivisionFilter:NoParamsForUpdate")
            return

        if likelihoodPeriodicGrid is not None:
            self.apd.gd.gridValues = self.apd.gd.gridValues * likelihoodPeriodicGrid

        if likelihoodsLinear is not None:
            # Update current linear distributions
            pass

        self.apd.gd = self.apd.gd.normalize(warnUnnorm=False)

    def get_estimate(self):
        return self.apd

    def get_point_estimate(self):
        return self.apd.hybridMean()

    def predict_linear(sysModel, sdGrid, sdSubDivGaussian):
        grid = sdGrid.grid
        sdHalfCondSdHalf = SdHalfCondSdHalfGridDistribution.fromFunction(
            lambda x, y: sysModel.transitionDensity(0, x, y),
            17,
            funDoesCartesianProduct=False,
            gridType='eq_point_set_symm',
            dim=2 * sysModel.dim
        )

        cartesian_product = np.array(np.meshgrid(range(grid.shape[1]), range(grid.shape[1]))).T.reshape(-1, 2)

        sdSubDivGaussianNew = []
        for i, j in cartesian_product:
            temp = sdSubDivGaussian[i].multiply(sdSubDivGaussian[j])
            temp = sdSubDivGaussian[i].multiply(sdHalfCondSdHalf.gridValues[j, i])
            sdSubDivGaussianNew.append(temp)
        sdSubDivGaussianNew = reduce(lambda x, y: x.add(y), sdSubDivGaussianNew)

        sdSubDivGaussianNew.gridValues = np.array([sdSubDivGaussianNew.gridValues[j, i] for i, j in cartesian_product])

        sdGridNew = sdGrid
        sdGridNew.gridValues = sdSubDivGaussianNew.gridValues

        return sdGridNew, sdSubDivGaussianNew
    
    def update(self, likelihood_periodic_grid, likelihoods_linear):
        if isinstance(likelihood_periodic_grid, AbstractDistribution):
            likelihood_periodic_grid = likelihood_periodic_grid.pdf(self.apd.gd.get_grid()).T

        assert (likelihood_periodic_grid.size == 0 or
                np.array_equal(likelihood_periodic_grid.shape, self.apd.gd.grid_values.shape))
        assert (likelihoods_linear.size == 0 or
                all([ld.dim == self.apd.linD for ld in likelihoods_linear]))

        assert len(likelihoods_linear) <= 1 or len(likelihoods_linear) == self.apd.gd.grid_values.shape[0]

        if likelihood_periodic_grid.size == 0 and likelihoods_linear.size == 0:
            print("Warning: Nothing to do for this update step.")
            return

        if likelihood_periodic_grid.size != 0:
            self.apd.gd.grid_values *= likelihood_periodic_grid

        if likelihoods_linear.size != 0:
            mu_preds = np.array([ld.mu for ld in self.apd.linear_distributions]).T
            mu_likelihoods = np.array([ld.mu for ld in likelihoods_linear]).T
            covs = np.dstack([ld.C for ld in self.apd.linear_distributions]) + np.dstack([ld.C for ld in likelihoods_linear])

            self.apd.gd.grid_values *= multivariate_normal.pdf(mu_preds.T, mu_likelihoods.T, covs)

            for i, ld in enumerate(self.apd.linear_distributions):
                j = i if len(likelihoods_linear) > 1 else 0
                C_est_inv_curr = np.linalg.inv(ld.C) + np.linalg.inv(likelihoods_linear[j].C)
                mu_est_curr = np.linalg.solve(C_est_inv_curr, np.dot(ld.C, ld.mu) + np.dot(likelihoods_linear[j].C, likelihoods_linear[j].mu))

                ld.mu = mu_est_curr
                ld.C = np.linalg.inv(C_est_inv_curr)

        self.apd.gd.normalize(warn_unnorm=False)
