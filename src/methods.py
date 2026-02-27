from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
import statsmodels.api as sm
from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd
from hyppo.conditional import PartialDcorr
from hyppo.independence import Hsic
from npeet.entropy_estimators import cmi
from tqdm.notebook import tqdm
import warnings

def rbf_kernel(X, sigma=None):
    """Compute RBF kernel matrix."""
    dists = squareform(pdist(X, 'euclidean'))
    sigma = np.median(dists[dists > 0])
    #sigma = np.sqrt(2.) * median_dist
    #sigma = np.sqrt(median_dist)
    return np.exp(-dists / (2 * sigma**2))


def kernel_distance_matrix(K):
    """Compute pairwise kernel distances from a kernel matrix."""
    diag_K = np.diag(K)
    D2 = diag_K[:, None] + diag_K[None, :] - 2 * K
    D2 = np.clip(D2, 0, None)  # numerical stability
    return np.sqrt(D2)

########## IMPLEMENTAZIONI DEI TEST STATISTICI ##########

def Kcondor_v2(r, z, x):
    n = r.shape[0]
    if n < 2: return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    kci_tester = KCI_CInd()
    #pval_kci, tstat_kci = kci_tester.kernel_matrix(data_x=z, data_y=r,
    kx,ky,kzx,kzy = kci_tester.kernel_matrix(data_x=r, data_y=z, data_z=x)
    _, kr_x, kz_x = kci_tester.KCI_V_statistic(kx,ky,kzx,kzy)
    Dr_x = kernel_distance_matrix(kr_x)
    Dz_x = kernel_distance_matrix(kz_x)
    Cr_x = fast_center(Dr_x)
    Cz_x = fast_center(Dz_x)
    dCov = lambda A, B: np.sum(A * B) / (n**2)
    dVar_r = dCov(Cr_x,Cr_x)
    dVar_z = dCov(Cz_x,Cz_x)
    return dCov(Cr_x, Cz_x) / np.sqrt(dVar_r * dVar_z + 1e-9)

def Kcondor_v2_opt(r, z, x):
    """Optimized version of Kcondor_v2 with Phase 1 improvements."""
    n = r.shape[0]
    if n < 2:
        return 0.0

    # Input validation
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)

    # KCI computation (unchanged)
    kci_tester = KCI_CInd()
    kx, ky, kzx, kzy = kci_tester.kernel_matrix(data_x=r, data_y=z, data_z=x)
    _, kr_x, kz_x = kci_tester.KCI_V_statistic(kx, ky, kzx, kzy)

    # Use optimized helper functions
    Dr_x = kernel_distance_matrix_opt(kr_x)
    Dz_x = kernel_distance_matrix_opt(kz_x)
    Cr_x = fast_center_opt(Dr_x)
    Cz_x = fast_center_opt(Dz_x)

    # Optimized distance covariance computation
    n_sq = n * n
    dVar_r = np.einsum('ij,ij->', Cr_x, Cr_x) / n_sq
    dVar_z = np.einsum('ij,ij->', Cz_x, Cz_x) / n_sq
    dCov_rz = np.einsum('ij,ij->', Cr_x, Cz_x) / n_sq

    return dCov_rz / np.sqrt(dVar_r * dVar_z + 1e-9)

def Kcondor_v2_opt2(r, z, x):
    """Second round optimization: exploits symmetry and fused operations."""
    n = r.shape[0]
    if n < 2:
        return 0.0

    # Input validation
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)

    # KCI computation (unchanged)
    kci_tester = KCI_CInd()
    kx, ky, kzx, kzy = kci_tester.kernel_matrix(data_x=r, data_y=z, data_z=x)
    _, kr_x, kz_x = kci_tester.KCI_V_statistic(kx, ky, kzx, kzy)

    # Use second-round optimized helper functions
    Dr_x = kernel_distance_matrix_opt2(kr_x)
    Dz_x = kernel_distance_matrix_opt2(kz_x)
    Cr_x = fast_center_opt2(Dr_x)
    Cz_x = fast_center_opt2(Dz_x)

    # Optimized distance covariance: pre-compute inverse and use multiplication
    n_sq_inv = 1.0 / (n * n)
    dVar_r = np.sum(Cr_x * Cr_x) * n_sq_inv
    dVar_z = np.sum(Cz_x * Cz_x) * n_sq_inv
    dCov_rz = np.sum(Cr_x * Cz_x) * n_sq_inv

    return dCov_rz / np.sqrt(dVar_r * dVar_z + 1e-9)

def Kcondor_v3(r, z, x):
    n = r.shape[0]
    if n < 2: return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    kci_tester = KCI_CInd()
    #pval_kci, tstat_kci = kci_tester.kernel_matrix(data_x=z, data_y=r,
    kx,ky,kzx,kzy = kci_tester.kernel_matrix(data_x=r, data_y=z, data_z=x)
    _, kr_x, kz_x = kci_tester.KCI_V_statistic(kx,ky,kzx,kzy)
    Dr_x = kernel_distance_matrix(kr_x)
    Dz_x = kernel_distance_matrix(kz_x)
    Cr_x = fast_center(Dr_x)
    Cz_x = fast_center(Dz_x)
    Kx = rbf_kernel(x)
    Dx = kernel_distance_matrix(Kx)
    Cx = fast_center(Dx)
    dCov = lambda A, B: np.sum(A * B) / (n**2)
    dVar_r = dCov(Cr_x,Cr_x)
    dVar_z = dCov(Cz_x,Cz_x)
    dVar_x = dCov(Cx,Cx)
    eps = 1e-9
    dCor_rz = dCov(Cr_x, Cz_x) / np.sqrt(dVar_r * dVar_z + eps)
    dCor_rx = dCov(Cr_x, Cx) / np.sqrt(dVar_r * dVar_x + eps)
    dCor_zx = dCov(Cz_x, Cx) / np.sqrt(dVar_z * dVar_x + eps)
    # Kernelized CONDOR
    num = dCor_rz - dCor_rx * dCor_zx
    den = np.sqrt(np.clip(1 - dCor_rx**2, 0, 1) * np.clip(1 - dCor_zx**2, 0, 1) + eps)
    return np.abs(num / den)

def fast_center(dist_r):
    """
    Center a square matrix using the centering formula:
    Dr = dist_r - row_mean - col_mean + total_mean
    Parameters
    ----------
    dist_r : np.ndarray
        Square matrix of shape (n, n)
    Returns
    -------
    Dr : np.ndarray
        Centered matrix of shape (n, n)
    """
    if dist_r.shape[0] != dist_r.shape[1]:
        raise ValueError("Input matrix must be square")
    row_mean = dist_r.mean(axis=1, keepdims=True)
    col_mean = dist_r.mean(axis=0, keepdims=True)
    total_mean = dist_r.mean()
    Dr = dist_r - row_mean - col_mean + total_mean
    return Dr


########## OPTIMIZED VERSIONS ##########

def kernel_distance_matrix_opt(K):
    """Optimized version: uses direct assignment instead of clip."""
    diag_K = np.diag(K)
    D2 = diag_K[:, None] + diag_K[None, :] - 2 * K
    D2[D2 < 0] = 0  # Faster than np.clip for lower bound only
    return np.sqrt(D2)


def fast_center_opt(dist_r):
    """Optimized version: uses sum + pre-computed divisions."""
    if dist_r.shape[0] != dist_r.shape[1]:
        raise ValueError("Input matrix must be square")

    n = dist_r.shape[0]
    n_inv = 1.0 / n
    n_sq_inv = n_inv * n_inv

    row_sum = dist_r.sum(axis=1, keepdims=True)
    col_sum = dist_r.sum(axis=0, keepdims=True)
    total_sum = row_sum.sum()

    row_mean = row_sum * n_inv
    col_mean = col_sum * n_inv
    total_mean = total_sum * n_sq_inv

    return dist_r - row_mean - col_mean + total_mean


########## THIRD ROUND: PRE-COMPUTATION FOR CALIBRATION ##########

class Kcondor_Precomputed:
    """
    Pre-computes fixed components (R_X and residualized r) for efficient calibration.

    This class exploits the fact that during CRT calibration, r and x are fixed
    while only z changes across B iterations. By pre-computing the expensive
    operations that depend only on r and x, we avoid O(n³) matrix operations
    in each iteration.

    Key optimization: Cache the residualizer matrix Rz (which requires matrix
    inversion) and reuse it to manually residualize kernels.
    """

    def __init__(self, r, x):
        """
        Pre-compute all components that depend on r and x.

        Parameters
        ----------
        r : np.ndarray
            Ranking/response variable (fixed during calibration)
        x : np.ndarray
            Task attributes/covariates (fixed during calibration)
        """
        from causallearn.utils.KCI.Kernel import Kernel as KCIKernel

        self.n = r.shape[0]
        if self.n < 2:
            self.valid = False
            return
        self.valid = True

        # Input validation
        if r.ndim == 1: r = r.reshape(-1, 1)
        if x.ndim == 1: x = x.reshape(-1, 1)

        # Store for later
        self.r = r
        self.x = x

        # Initialize KCI tester to get epsilon values
        self.kci_tester = KCI_CInd()
        epsilon = self.kci_tester.epsilon_x

        # Pre-compute kernels for r and x (these never change)
        dummy_z = np.zeros_like(r)
        kr, _, kzx, kzy = self.kci_tester.kernel_matrix(data_x=r, data_y=dummy_z, data_z=x)

        # **KEY OPTIMIZATION**: Pre-compute the residualizer matrix Rzx once
        # This is the expensive O(n³) operation (matrix inversion via pinv)
        # Rzx = epsilon * pinv(Kzx + epsilon * I)
        _, self.Rzx = KCIKernel.center_kernel_matrix_regression(kr, kzx, epsilon)

        # Pre-compute residualized r kernel using cached Rzx
        # kr_x = Rzx @ Kr @ Rzx (this is fast, just matrix multiplication)
        self.kr_x_fixed = self.Rzx.dot(kr.dot(self.Rzx))

        # Pre-compute distance and centered matrices for r
        self.Dr_x_fixed = kernel_distance_matrix_opt2(self.kr_x_fixed)
        self.Cr_x_fixed = fast_center_opt2(self.Dr_x_fixed)

        # Pre-compute variance of r (never changes)
        n_sq_inv = 1.0 / (self.n * self.n)
        self.dVar_r_fixed = np.sum(self.Cr_x_fixed * self.Cr_x_fixed) * n_sq_inv
        self.n_sq_inv = n_sq_inv

    def score(self, z):
        """
        Compute Kcondor score with pre-computed r and x components.

        This method only computes the parts that depend on z, which changes
        in each calibration iteration.

        **KEY**: We manually apply the cached Rzx matrix instead of calling
        KCI_V_statistic, avoiding the expensive matrix inversion.

        Parameters
        ----------
        z : np.ndarray
            The variable to test (changes during calibration)

        Returns
        -------
        float
            The Kcondor test statistic
        """
        if not self.valid:
            return 0.0

        # Input validation
        if z.ndim == 1: z = z.reshape(-1, 1)

        # Compute kernel for z (only thing that changes)
        _, kz, _, _ = self.kci_tester.kernel_matrix(
            data_x=self.r, data_y=z, data_z=self.x
        )

        # **KEY OPTIMIZATION**: Manually apply the cached residualizer Rzx
        # Instead of calling KCI_V_statistic (which recomputes Rzx via pinv),
        # we directly compute: kz_x = Rzx @ Kz @ Rzx
        # This is O(n²) instead of O(n³)!
        kz_x = self.Rzx.dot(kz.dot(self.Rzx))

        # Convert to distance and center
        Dz_x = kernel_distance_matrix_opt2(kz_x)
        Cz_x = fast_center_opt2(Dz_x)

        # Compute distance covariance components (using pre-computed dVar_r)
        dVar_z = np.sum(Cz_x * Cz_x) * self.n_sq_inv
        dCov_rz = np.sum(self.Cr_x_fixed * Cz_x) * self.n_sq_inv

        return dCov_rz / np.sqrt(self.dVar_r_fixed * dVar_z + 1e-9)


def Kcondor_v2_precomputed_factory(r, x):
    """
    Factory function to create a scoring function with pre-computed r and x.

    This is designed to be used with the calibration loop where r and x are fixed.

    Parameters
    ----------
    r : np.ndarray
        Fixed ranking
    x : np.ndarray
        Fixed covariates

    Returns
    -------
    callable
        A function that takes only z as argument

    Example
    -------
    >>> scoring_fn = Kcondor_v2_precomputed_factory(r_test, x_test)
    >>> # Now use in calibration loop where only z changes
    >>> for _ in range(B):
    >>>     z_new = generator.sample(x_test)
    >>>     score = scoring_fn(z_new)
    """
    precomputed = Kcondor_Precomputed(r, x)
    return precomputed.score


########## SECOND ROUND OPTIMIZATIONS ##########

def kernel_distance_matrix_opt2(K):
    """Second optimization: fuses maximum and sqrt operations."""
    diag_K = np.diag(K)
    D2 = diag_K[:, None] + diag_K[None, :] - 2 * K
    # Fuse clipping and sqrt for better performance
    return np.sqrt(np.maximum(D2, 0))


def fast_center_opt2(dist_r):
    """Second optimization: exploits symmetry of distance matrices."""
    if dist_r.shape[0] != dist_r.shape[1]:
        raise ValueError("Input matrix must be square")

    n = dist_r.shape[0]
    n_inv = 1.0 / n
    n_sq_inv = n_inv * n_inv

    # Distance matrices are symmetric, so row_sum.T == col_sum
    # Only compute once and reuse
    row_sum = dist_r.sum(axis=1, keepdims=True)
    total_sum = row_sum.sum()

    row_mean = row_sum * n_inv
    col_mean = row_mean.T  # Exploit symmetry instead of recomputing
    total_mean = total_sum * n_sq_inv

    return dist_r - row_mean - col_mean + total_mean




def kcondor_score(r, z, x, sigma=0.01):
    """Kernelized CONDOR score using RBF kernel."""
    n = r.shape[0]
    if n < 2: 
        return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    
    # Kernel matrices
    Kr = rbf_kernel(r, sigma)
    Kz = rbf_kernel(z, sigma)
    Kx = rbf_kernel(x, sigma)
    
    # Convert to kernel distance matrices
    Dr = kernel_distance_matrix(Kr)
    Dz = kernel_distance_matrix(Kz)
    Dx = kernel_distance_matrix(Kx)
    
    # Centering
    H = np.eye(n) - 1/n
    Dr_c, Dz_c, Dx_c = H @ Dr @ H, H @ Dz @ H, H @ Dx @ H
    
    # Distance covariance and correlation
    dCov = lambda A, B: np.sum(A * B) / (n**2)
    dVar_r, dVar_z, dVar_x = dCov(Dr_c, Dr_c), dCov(Dz_c, Dz_c), dCov(Dx_c, Dx_c)
    eps = 1e-9
    dCor_rz = dCov(Dr_c, Dz_c) / np.sqrt(dVar_r * dVar_z + eps)
    dCor_rx = dCov(Dr_c, Dx_c) / np.sqrt(dVar_r * dVar_x + eps)
    dCor_zx = dCov(Dz_c, Dx_c) / np.sqrt(dVar_z * dVar_x + eps)
    
    # Kernelized CONDOR
    num = dCor_rz - dCor_rx * dCor_zx
    den = np.sqrt(np.clip(1 - dCor_rx**2, 0, 1) * np.clip(1 - dCor_zx**2, 0, 1) + eps)
    return np.abs(num / den)

def pdnhsic_v2(r, z, x):
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    kci_tester = KCI_UInd() #Unconditional i.e. Hsic
    rz_r, rz_z = kci_tester.kernel_matrix(r, z)
    rx_r, rx_x = kci_tester.kernel_matrix(r, x)
    zx_z, zx_x = kci_tester.kernel_matrix(z, x)
    nh_rz, _, _ = kci_tester.HSIC_V_statistic(rz_r, rz_z)
    nh_rx, _, _ = kci_tester.HSIC_V_statistic(rx_r, rx_x)
    nh_zx, _, _ = kci_tester.HSIC_V_statistic(zx_z, zx_x)
    num = nh_rz - nh_rx * nh_zx
    den = np.sqrt(np.clip(1 - nh_rx**2, 0, 1) * np.clip(1 - nh_zx**2, 0, 1) + 1e-9)
    return np.abs(num / den)



def pdnhsic_old(r, z, x, sigma=0.01):
    """Optimized PASₙHSIC computation (single-pass style)."""
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    n = r.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    Kr, Kz, Kx = rbf_kernel(r, sigma), rbf_kernel(z, sigma), rbf_kernel(x, sigma)

    # Center once to save ops
    HKrH = H @ Kr @ H
    HKzH = H @ Kz @ H
    HKxH = H @ Kx @ H

    # Efficient normalized HSIC computation
    def nh(KA, KB):
        num = np.trace(KA @ KB)
        den = np.sqrt(np.trace(KA @ KA) * np.trace(KB @ KB)) + 1e-9
        return num / den

    nh_rz = nh(HKrH, HKzH)
    nh_rx = nh(HKrH, HKxH)
    nh_zx = nh(HKzH, HKxH)

    num = nh_rz - nh_rx * nh_zx
    den = np.sqrt(np.clip(1 - nh_rx**2, 0, 1) * np.clip(1 - nh_zx**2, 0, 1) + 1e-9)
    return np.abs(num / den)






def nhsic(r, z, x):
    n = r.shape[0]
    if n < 2: return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    kci_tester = KCI_UInd()
    pval, test_s = kci_tester.compute_pvalue(data_x=z, data_y=r)
    #return test_s
    return pval

######## IMPLEMENTAZIONI DEI TEST STATISTICI OTHERS ##########



def condor_score(r, z, x):
    n = r.shape[0]
    if n < 2: return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    dist_r, dist_z, dist_x = squareform(pdist(r)), squareform(pdist(z)), squareform(pdist(x))
    H = np.eye(n) - 1/n
    D_r_c, D_z_c, D_x_c = H @ dist_r @ H, H @ dist_z @ H, H @ dist_x @ H
    dCov = lambda A, B: np.sum(A * B) / (n**2)
    dVar_r, dVar_z, dVar_x = dCov(D_r_c, D_r_c), dCov(D_z_c, D_z_c), dCov(D_x_c, D_x_c)
    eps = 1e-9

    # FIX 5: Aggiunto controllo di stabilità numerica. Se la varianza è quasi zero, la correlazione è 0.
    if dVar_r < eps or dVar_z < eps or dVar_x < eps:
        return 0.0

    dCor_rz = dCov(D_r_c, D_z_c) / np.sqrt(dVar_r * dVar_z + eps)
    dCor_rx = dCov(D_r_c, D_x_c) / np.sqrt(dVar_r * dVar_x + eps)
    dCor_zx = dCov(D_z_c, D_x_c) / np.sqrt(dVar_z * dVar_x + eps)
    num = dCor_rz - dCor_rx * dCor_zx
    den = np.sqrt(np.clip(1 - dCor_rx**2, 0, 1) * np.clip(1 - dCor_zx**2, 0, 1) + eps)
    return np.abs(num / den)




def nkci_score(r, z, x):
    n = r.shape[0]
    if n < 2: return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    kci_tester = KCI_CInd()
    kx,ky,kzx,kzy = kci_tester.kernel_matrix(data_x=z, data_y=r, data_z=x)
    tstat_kci, kxr, kyr = kci_tester.KCI_V_statistic(kx,ky,kzx,kzy)
    return tstat_kci

def kci_pval(r, z, x):
    n = r.shape[0]
    if n < 2: return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    if x.ndim == 1: x = x.reshape(-1, 1)
    kci_tester = KCI_CInd()
    #pval_kci, tstat_kci = kci_tester.kernel_matrix(data_x=z, data_y=r,
    pval, _ = kci_tester.compute_pvalue(data_x=z, data_y=r, data_z=x)
    return pval


def nhsic_score(r, z, x):
    n = r.shape[0]
    if n < 2: return 0.0
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    sig = lambda V: np.median(pdist(V)) + 1e-9
    K_r, K_z = rbf_kernel(r, sig(r)), rbf_kernel(z, sig(z))
    H = np.eye(n) - 1/n
    num = np.trace(K_r @ H @ K_z @ H)
    den_r_sq, den_z_sq = np.trace(K_r @ H @ K_r @ H), np.trace(K_z @ H @ K_z @ H)
    if den_r_sq < 1e-9 or den_z_sq < 1e-9: return 0.0
    return max(0, num / np.sqrt(den_r_sq * den_z_sq))

def cmi_score(r, z, x):
    """Wrapper per lo stimatore di Conditional Mutual Information di NPEET."""
    if r.ndim == 1: r = r.reshape(-1, 1)
    # Calcola I(Z; R | X)
    return cmi(z, r, x)

def hsic_hyppo_score(r, z, x):
    if r.ndim == 1: r = r.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    
    r_64 = r.astype(np.float64)
    z_64 = z.astype(np.float64)
    stat = Hsic().statistic(r_64, z_64) 
    return stat

def partial_dcorr_score(r, z, x):
    if r.ndim == 1: r = r.reshape(-1, 1)
    
    r_64 = r.astype(np.float64)
    z_64 = z.astype(np.float64)
    x_64 = x.astype(np.float64)
    
    # Check for zero variance in z
    if np.var(z_64) == 0:
        # Add small noise to avoid zero variance
        z_64 = z_64 + np.random.normal(0, 1e-6, z_64.shape)
    
    #stat = PartialDcorr().statistic(r_64, z_64, x_64)
    #return stat
    try:
        stat,pval  = PartialDcorr().test(r_64, z_64, x_64)
        return pval
    except ValueError as e:
        if "0 variance" in str(e):
            # Return p-value of 1.0 if variance check fails
            return 1.0
        else:
            raise


def _design_with_constant(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.ones((matrix.shape[0], 1), dtype=np.float64)
    return sm.add_constant(matrix, has_constant="add")


def partial_corr_pg_score(r, z, x) -> float:
    """Partial correlation score via incremental R-squared from OLS models."""
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    r_64 = r.astype(np.float64)
    z_64 = z.astype(np.float64)
    x_64 = x.astype(np.float64)

    n = r_64.shape[0]
    y = r_64.squeeze()
    eps = 1e-9

    if np.var(y) < eps or z_64.shape[1] == 0:
        return 0.0

    if np.all(np.var(z_64, axis=0) < eps):
        return 0.0

    X_cov = x_64 if x_64.size else np.empty((n, 0))

    try:
        X_reduced = _design_with_constant(X_cov)
        X_full = _design_with_constant(np.hstack([X_cov, z_64]))

        model_reduced = sm.OLS(y, X_reduced).fit()
        model_full = sm.OLS(y, X_full).fit()
    except Exception:
        return 0.0

    R_reduced = float(model_reduced.rsquared) if model_reduced.rsquared is not None else 0.0
    R_full = float(model_full.rsquared) if model_full.rsquared is not None else 0.0

    if np.isnan(R_reduced) or np.isnan(R_full):
        return 0.0

    if R_full < R_reduced:
        R_full = R_reduced

    denom = max(eps, 1.0 - R_reduced)
    partial_R_sq = np.clip((R_full - R_reduced) / denom, 0.0, 1.0)
    return float(np.sqrt(partial_R_sq))