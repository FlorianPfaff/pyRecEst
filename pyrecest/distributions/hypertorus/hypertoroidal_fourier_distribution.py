import numpy as np
from numpy import pi
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from .hypertoroidal_uniform_distribution import HypertoroidalUniformDistribution
import copy
import scipy
from ..abstract_orthogonal_basis_distribution import AbstractOrthogonalBasisDistribution
import warnings

class HypertoroidalFourierDistribution(AbstractHypertoroidalDistribution, AbstractOrthogonalBasisDistribution):
    def __init__(self, coeff_mat, transformation='sqrt'):
        if isinstance(coeff_mat, AbstractHypertoroidalDistribution):
            raise ValueError('fourierCoefficients:invalidCoefficientMatrix', 'You gave a distribution as the first argument. To convert distributions to a distribution in Fourier representation, use .fromDistribution.')
        elif np.ndim(coeff_mat) == 0:
            print('fourierCoefficients:singleCoefficient', 'Fourier series only has one element, assuming dimension 1.')
            coeff_mat = np.atleast_1d(coeff_mat)
      
        AbstractHypertoroidalDistribution.__init__(self, np.ndim(coeff_mat))
        AbstractOrthogonalBasisDistribution.__init__(self, coeff_mat, transformation=transformation)

    def value(self, xs):
        assert xs.shape[-1] == self.dim or xs.ndim == 1 and self.dim == 1
        assert all((np.array(self.coeff_mat.shape)-1) % 2 == 0), "Supporting even numbers using complex coefficients would make it indistinguishable from the next higher number of coefficients. Therefore, it is not supported."
        maxk = (np.array(self.coeff_mat.shape) - 1) // 2
        kRanges = [range(-currMaxk, currMaxk+1) for currMaxk in maxk]
        individualIndexMatrices = np.meshgrid(*kRanges, indexing='ij')

        # Reshape individualIndexMatrices and xs for broadcasting
        individualIndexMatrices = np.stack(individualIndexMatrices).reshape(self.dim, -1)
        xs = xs[..., None] # Add an extra dimension for broadcasting
        
        matCurr = individualIndexMatrices * xs
        if self.dim > 1: # To prevent summing over all input values for dim
            matCurr = np.sum(matCurr, axis=-2) # Sum over dimension representing xs's elements
        matCurr = self.coeff_mat.flatten() * np.exp(matCurr * 1j)
        
        p = np.sum(matCurr, axis=-1) # Sum over the last dimension
        assert np.all(np.imag(p) < 0.1)
        return np.real(p)
    
    def trigonometric_moment(self, n):
        if n == 0:
            return np.ones(self.dim)
        
        elif n < 0:
            return np.conj(self.trigonometric_moment(-n))
        
        else:
            if self.transformation == 'sqrt':
                tfd_tmp = self.transform_via_coefficients('square', (2 * n * np.ones(self.dim) + 1).astype(int))
            
            elif self.transformation == 'identity':
                tfd_tmp = self.truncate((2 * n * np.ones(self.dim) + 1).astype(int))
            
            else:
                raise ValueError('Transformation not recognized or unsupported')
            
            indices = (n + 1) * np.ones(self.dim, dtype=int) - n * np.eye(self.dim, dtype=int)
            m = (2 * np.pi) ** self.dim * tfd_tmp.coeff_mat[tuple(indices)]
            
            return m

    def multiply(self, f2, n_coefficients=None):
        # Check for transformation compatibility
        if self.transformation != f2.transformation:
            raise ValueError("Transformations must match.")
        
        if n_coefficients is None:
            n_coefficients = self.C.shape
            
        
        # Perform multiplication based on transformation
        if self.transformation == 'log':
            f = copy.deepcopy(self)  # Replace with your own copy method
            if n_coefficients != f2.coeff_mat.shape:
                self.truncate(n_coefficients)
            if n_coefficients != f2.coeff_mat.shape:
                f2.truncate(n_coefficients)
            f.coeff_mat = self.coeff_mat + f2.coeff_mat
            # Log-transformed, not normalizing
            print("Not performing normalization when using log transformation.")
        
        elif self.transformation in ['identity', 'sqrt']:
            f = copy.deepcopy(self)
            f.coeff_mat = scipy.ndimage.convolve(self.coeff_mat, f2.coeff_mat, 'constant', cval=0) 
        else:
            raise ValueError("Transformation not recognized or unsupported.")
        
        # Truncate and normalize
        f.truncate(n_coefficients, True)
        
        return f
    
    def transform_via_coefficients(self, desired_transformation, no_of_coefficients=None):
        if no_of_coefficients is None:
            no_of_coefficients = [self.C.shape[i] for i in range(self.dim)]

        if desired_transformation == 'identity':
            return self  # If the desired transformation is identity, just return self

        elif desired_transformation == 'square':
            if self.transformation == 'sqrt':
                new_trans = 'identity'
            elif self.transformation == 'identity':
                new_trans = 'square'
            else:
                new_trans = 'multiple'

            # Create a new instance to store the transformed coefficients
            transformed_instance = HypertoroidalFourierDistribution(self.C, new_trans)
            
            # Perform the transformation (square)
            transformed_instance.C = scipy.ndimage.convolve(self.coeff_mat, self.coeff_mat, 'constant', cval=0) 

            transformed_instance.transformation = new_trans
            transformed_instance = transformed_instance.truncate(no_of_coefficients)
            transformed_instance = transformed_instance.normalize()

            return transformed_instance

        else:
            raise ValueError('Desired transformation not supported via coefficients')

    def integrate(self, integration_boundaries):
        if integration_boundaries is None:
            left = np.zeros(self.dim)
            right = 2 * pi * np.ones(self.dim)
        raise NotImplementedError()

    def truncate(self, n_coefficients, forceNormalization=False):
        # Truncates Fourier series. Fills up if there are less coefficients
        # Expects number of complex coefficients (or sum of number of real
        # coefficients)

        if np.size(n_coefficients) == 1:
            n_coefficients = n_coefficients * np.ones(self.dim)
        n_coefficients = np.asarray(n_coefficients)

        assert all((n_coefficients - 1) % 2 == 0) and (all(n_coefficients > 1) or self.dim == 1), \
            'Invalid number of coefficients, numbers for all dimensions have to be odd and greater than 1.'

        result = copy.deepcopy(self)

        if np.array_equal(result.coeff_mat.shape, n_coefficients):  # Do not need to truncate as already correct size
            if not forceNormalization:
                return result
            else:
                return result.normalize(warn_unnorm=False)

        elif any(result.coeff_mat.shape < n_coefficients):
            print("Warning: At least in one dimension, truncate has to fill up due to too few coefficients.")

        maxksold = (np.array(result.coeff_mat.shape) - 1) / 2
        maxksnew = (n_coefficients - 1) / 2
        coeff_mat_new = np.zeros(n_coefficients, dtype=complex)

        indicesNewMat = [list(range(int(max(maxksnew[i]-maxksold[i]+1, 1)), int(min(maxksnew[i]+maxksold[i]+1, coeff_mat_new.shape[i])))) for i in range(result.dim)]
        indicesOldMat = [list(range(int(max(maxksold[i]-maxksnew[i]+1, 1)), int(min(maxksold[i]+maxksnew[i]+1, result.coeff_mat.shape[i])))) for i in range(result.dim)]

        np.add.at(coeff_mat_new, tuple(indicesNewMat), result.coeff_mat[tuple(indicesOldMat)])

        result.coeff_mat = coeff_mat_new

        # Truncation can void normalization if transformation is not
        # identity. Normalize only if enforced by the user or if
        # truncation can void normalization.
        if forceNormalization or (self.transformation != 'identity' and any(n_coefficients < np.array(self.C.shape))):
            # Disable warning as we expect normalization to be
            # necessary
            return result.normalize(warn_unnorm=False)
        return result
    
    def normalize_in_place(self, tol=1e-4, warn_unnorm=True):
        # Normalize Fourier density while taking its type into account
        if self.transformation == 'sqrt':
            c00 = np.linalg.norm(self.coeff_mat)**2  # Square root calculated later to use norm and not squared norm
            factor_for_id = c00 * (2 * np.pi)**(self.dim)
            normalization_factor = np.sqrt(factor_for_id)
        elif self.transformation == 'identity':
            # This will always get the most central element that
            # corresponds to c00.
            center_indices = tuple(x//2 for x in self.coeff_mat.shape)
            c00 = self.coeff_mat[center_indices]
            factor_for_id = c00 * (2 * np.pi)**(self.dim)
            normalization_factor = factor_for_id
        else:
            warnings.warn('Warning: Unable to test if normalized')
            return self

        if c00 < 0:
            warnings.warn('Warning: C00 is negative. This can either be caused by a user error or due to negativity caused by non-square rooted version')
        elif np.abs(c00) < 1e-200:  # Tolerance has to be that low to avoid unnecessary errors on multiply
            raise ValueError('C00 is too close to zero, this usually points to a user error')
        elif np.abs(factor_for_id - 1) > tol:
            if warn_unnorm:
                warnings.warn('Warning: Coefficients apparently do not belong to normalized density. Normalizing...')
        else:
            return self  # Normalized, return original density

        self.C = self.coeff_mat / normalization_factor
        return self

    @staticmethod
    def from_function(fun, n_coefficients, dim, desired_transformation='sqrt'):
        # Dimension of noOfCoefficients has to be in accordance to
        # dimensionality of the function to approximate.
        if np.size(n_coefficients) == 1:
            n_coefficients = n_coefficients * np.ones(dim, dtype=int)
            
        assert np.size(n_coefficients) == dim
        assert (len(n_coefficients) == fun.__code__.co_argcount or fun.__code__.co_argcount == 0), \
            'n_coefficients has to match dimensionality (in form of number of input arguments) of the function.'

        gridIndividualAxis = [np.linspace(0, 2 * pi, currNo, endpoint=False) for currNo in n_coefficients]
        gridCell = np.meshgrid(*gridIndividualAxis)
        fvals = fun(*gridCell)  # Assume functions are vectorized!
        
        assert np.prod(np.shape(fvals)) == np.prod(n_coefficients), \
            'Size of output of pdf is incorrect. Please ensure that pdf returns only one scalar per dim-tupel.'
        
        return HypertoroidalFourierDistribution.from_function_values(fvals, n_coefficients, desired_transformation)
    
    def pdf(self, xs: np.ndarray):
        return AbstractOrthogonalBasisDistribution.pdf(self, xs)
    
    @classmethod
    def from_function_values(cls, fvals, n_coefficients=None, desired_transformation='sqrt', already_transformed=False):
        if n_coefficients is None:
            n_coefficients = np.array(fvals.shape)

        # Ensure that n (or 1) sizes are given for n-d tensors
        assert np.size(n_coefficients) == fvals.ndim or np.size(n_coefficients) == 1, "no_of_coefficients does not match the number of dimensions of fvals."

        # Creates Fourier distribution from function values
        # Assumes fvals are not yet transformed, use custom if they already
        # are transformed
        assert np.all((np.array(n_coefficients) - 1) % 2 == 0), "Invalid number of coefficients, numbers for all dimensions have to be odd."

        # Cannot directly compare the size of fvals with noOfCoefficients because we
        # allow truncation afterward.
        assert np.all(np.array(fvals.shape) > 1), "Some dimension only has one entry along it. Fix this."

        if not already_transformed:
            if desired_transformation == 'sqrt':
                fvals = np.sqrt(fvals)
            elif desired_transformation == 'log':
                fvals = np.log(fvals)
            elif desired_transformation == 'identity':
                pass  # keep them unchanged
            elif desired_transformation == 'custom':
                pass  # already transformed
            else:
                raise ValueError('Transformation not recognized or unsupported by transformation via FFT')

        fourier_coefficients = np.fft.fftshift(np.fft.fftn(fvals) / np.prod(fvals.shape))

        if not np.all(np.mod(fourier_coefficients.shape, 2) == 1):
            # Fill it up with the mirrored version if there are even
            # numbers
            fourier_coefficients = np.pad(fourier_coefficients, [(0, int(i == 0)) for i in np.mod(fourier_coefficients.shape, 2)], 'constant')

            indices_for_reversing = [range(i - 1, -1, -1) for i in fourier_coefficients.shape]
            fourier_coefficients = 0.5 * (fourier_coefficients + np.conj(fourier_coefficients[tuple(indices_for_reversing)]))

        hfd = cls(fourier_coefficients, desired_transformation)
        return hfd.truncate(n_coefficients)
    
    @classmethod
    def from_distribution(cls, distribution, n_coefficients, desired_transformation='sqrt'):
        # Check input arguments
        assert isinstance(n_coefficients, int) or all(isinstance(i, int) for i in n_coefficients), \
            'no_of_coefficients must be a integers'
        assert np.all(np.array(n_coefficients) > 0), "no_of_coefficients must be positive"
        assert isinstance(n_coefficients, int) and n_coefficients % 2 == 1 or all([nc % 2 == 1 for nc in n_coefficients]), \
            'no_of_coefficients must be odd'

        # Handle distribution types
        if isinstance(distribution, HypertoroidalUniformDistribution):
            C = np.zeros(n_coefficients) 
            if desired_transformation == 'sqrt':
                C = 1 / np.sqrt((2 * np.pi) ** distribution.dim)
            elif desired_transformation == 'identity':
                C = 1 / (2 * np.pi) ** distribution.dim
            else:
                raise ValueError('Transformation not recognized or unsupported')
            hfd = cls(C, desired_transformation)
        elif not isinstance(distribution, AbstractHypertoroidalDistribution):
            raise ValueError('Invalid distribution type')
        
        hfd = cls.from_function(lambda *args: np.reshape(distribution.pdf(np.column_stack([arg.flatten() for arg in args])), args[0].shape), n_coefficients = n_coefficients, dim = distribution.dim)
        
        return hfd
