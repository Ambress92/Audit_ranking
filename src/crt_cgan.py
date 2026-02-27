import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs

from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt

# --- Conditional GAN Implementation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, x_dim, z_dim, noise_dim, hidden_dim, z_type):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + noise_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        if z_type == 'binary':
            self.net.add_module("sigmoid", nn.Sigmoid())
    def forward(self, x, noise):
        return self.net(torch.cat([x, noise], dim=1))

class Discriminator(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=1))

class ConditionalGAN:
    def __init__(self, x_dim, z_dim, noise_dim=16, hidden_dim=64, epochs=30, batch_size=64, lr=2e-4):
        self.x_dim, self.z_dim, self.noise_dim = x_dim, z_dim, noise_dim
        self.hidden_dim = hidden_dim
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.z_type = 'continuous'
        self.x_scaler, self.z_scaler = StandardScaler(), StandardScaler()

    def fit(self, x_train, z_train):
        if np.all(np.isin(z_train, [0, 1])): self.z_type = 'binary'
        x_scaled = self.x_scaler.fit_transform(x_train)
        z_scaled = self.z_scaler.fit_transform(z_train) if self.z_type == 'continuous' else z_train
        
        self.G = Generator(self.x_dim, self.z_dim, self.noise_dim, self.hidden_dim, self.z_type).to(device)
        self.D = Discriminator(self.x_dim, self.z_dim, self.hidden_dim).to(device)
        opt_g, opt_d = optim.Adam(self.G.parameters(), lr=self.lr), optim.Adam(self.D.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        
        loader = DataLoader(TensorDataset(torch.tensor(x_scaled, dtype=torch.float32), 
                                          torch.tensor(z_scaled, dtype=torch.float32)), 
                            batch_size=self.batch_size, shuffle=True)
        # NOTA 7: Per un'implementazione production-grade, si potrebbe aggiungere un meccanismo di early stopping.
        # Per questo esempio, un numero fisso di epoche è sufficiente.
        for epoch in tqdm(range(self.epochs), desc="GAN Training", leave=False):
            for xb, zb in loader:
                xb, zb = xb.to(device), zb.to(device)
                opt_d.zero_grad()
                noise = torch.randn(xb.size(0), self.noise_dim, device=device)
                fake_z = self.G(xb, noise)
                real_out, fake_out = self.D(xb, zb), self.D(xb, fake_z.detach())
                loss_d = loss_fn(real_out, torch.ones_like(real_out)) + loss_fn(fake_out, torch.zeros_like(fake_out))
                loss_d.backward(); opt_d.step()
                opt_g.zero_grad()
                gen_out = self.D(xb, fake_z)
                loss_g = loss_fn(gen_out, torch.ones_like(gen_out))
                loss_g.backward(); opt_g.step()

    def sample(self, x_test):
        self.G.eval()
        with torch.no_grad():
            x_scaled = self.x_scaler.transform(x_test)
            noise = torch.randn(x_test.shape[0], self.noise_dim, device=device)
            fake_z_scaled = self.G(torch.tensor(x_scaled, dtype=torch.float32).to(device), noise).cpu().numpy()
        if self.z_type == 'binary':
            return (fake_z_scaled > 0.5).astype(int)
        else:
            return self.z_scaler.inverse_transform(fake_z_scaled)

# --- Main CRT Calibration Function ---

def crt_calibration_efficient(x, z, r, scoring_function, kf_splits, trained_generators, B=500):
    fold_p_values = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf_splits):
        x_test, z_test, r_test = x[test_idx], z[test_idx], r[test_idx]
        generator = trained_generators[fold_idx]
        score_obs = scoring_function(r_test, z_test, x_test)
        null_dist = []
        for _ in range(B):
            try:
                z_gen = generator.sample(x_test)
                # Check for zero variance
                if np.var(z_gen) == 0:
                    # Add small noise to avoid zero variance
                    z_gen = z_gen + np.random.normal(0, 1e-6, z_gen.shape)
                score = scoring_function(r_test, z_gen, x_test)
                null_dist.append(score)
            except ValueError as e:
                if "0 variance" in str(e):
                    # Skip this sample
                    continue
                else:
                    raise
        if len(null_dist) == 0:
            # If all samples failed, use p-value of 1.0 (most conservative)
            p_k = 1.0
        else:
            m_k = np.sum(np.abs(null_dist) >= np.abs(score_obs))
            p_k = m_k / len(null_dist)
        fold_p_values.append(p_k)
    #fisher_stat = -2 * np.sum(np.log(np.clip(fold_p_values, 1e-10, 1.0)))
    #p_final = chi2.sf(fisher_stat, df=2 * len(kf_splits))
    p_final = np.mean(fold_p_values)
    #print(f"--> P-value finale per '{scoring_function.__name__}': {p_final:.6f}")
    #print(f"    P-values per fold: {np.round(fold_p_values, 4)}")
    return p_final


def crt_calibration_precomputed(x, z, r, scoring_factory, kf_splits, trained_generators, B=500):
    """
    Optimized CRT calibration that pre-computes fixed components.

    This function is designed for scoring functions that can be pre-computed
    based on fixed r and x, then efficiently evaluated for different z values.

    The key optimization is that during the calibration loop, r and x are fixed
    while only z changes. By pre-computing the expensive operations (like matrix
    inversions) that depend on r and x, we avoid repeating O(n³) operations B times.

    Parameters
    ----------
    x : np.ndarray
        Task attributes/covariates
    z : np.ndarray
        Variable to test for conditional independence
    r : np.ndarray
        Rankings/responses
    scoring_factory : callable
        A factory function that takes (r, x) and returns a function that takes only z
        Example: methods.Kcondor_v2_precomputed_factory
    kf_splits : list of tuples
        K-fold cross-validation splits
    trained_generators : list
        Pre-trained conditional generators for each fold
    B : int, optional
        Number of bootstrap samples (default: 500)

    Returns
    -------
    float
        Combined p-value across folds

    Performance Notes
    ----------------
    - Original: O(B * n³) per fold (B matrix inversions)
    - Optimized: O(n³ + B * n²) per fold (1 matrix inversion + B matrix multiplications)
    - Expected speedup: ~B times faster for the KCI-dependent operations
    """
    fold_p_values = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf_splits):
        x_test, z_test, r_test = x[test_idx], z[test_idx], r[test_idx]
        generator = trained_generators[fold_idx]

        # Pre-compute the scoring function with fixed r and x
        # This does the expensive O(n³) matrix inversion once
        scoring_fn = scoring_factory(r_test, x_test)

        # Compute observed score (uses the precomputed components)
        score_obs = scoring_fn(z_test)

        # Generate null distribution - now each iteration is O(n²) instead of O(n³)
        null_dist = [scoring_fn(generator.sample(x_test)) for _ in range(B)]

        m_k = np.sum(np.abs(null_dist) >= np.abs(score_obs))
        p_k = m_k / B
        fold_p_values.append(p_k)

    p_final = np.mean(fold_p_values)
    return p_final