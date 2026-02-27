import numpy as np
import random

from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import chi2
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_blobs
from scipy.linalg import inv
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
#from ucimlrepo import fetch_ucirepo
from tqdm.notebook import tqdm



# --- Synthetic Data Generation Functions ---
def generate_ranking_data(
    n=1000, d_o=5, d_p=1,
    kx=3, kt=2,
    gamma=0.5,
    sigma_eps=0.0,
    random_state=None
):
    rng = np.random.default_rng(random_state)
    X, _ = make_blobs(n_samples=n, n_features=d_o, centers=kx, cluster_std=1.0, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    T, _ = make_blobs(n_samples=n, n_features=d_p, centers=kt, cluster_std=1.0, random_state=random_state+1 if random_state is not None else None)
    T = StandardScaler().fit_transform(T)
    raw = np.ones((d_o, d_p))
    alpha = gamma * raw / raw.sum(axis=0, keepdims=True)
    Z = (1 - gamma) * T + X @ alpha + sigma_eps * rng.normal(size=(n, d_p))
    X = MinMaxScaler().fit_transform(X)
    Z = MinMaxScaler().fit_transform(Z)
    return X, Z, alpha

def generate_ranking_data_reverse(
    n=1000, d_o=5, d_p=1,
    kx=3, kt=2,
    gamma=0.5,
    sigma_eps=0.0,
    random_state=None
):
    """Generate X from Z instead of Z from X.
    """
    rng = np.random.default_rng(random_state)
    Z, _ = make_blobs(n_samples=n, n_features=d_p, centers=kx, cluster_std=1.0, random_state=random_state)
    Z = StandardScaler().fit_transform(Z)
    # Generate X from Z
    T, _ = make_blobs(n_samples=n, n_features=d_o, centers=kt, cluster_std=1.0, random_state=random_state+1 if random_state is not None else None)
    T = StandardScaler().fit_transform(T)
    raw = np.ones((d_p, d_o))
    alpha = gamma * raw / raw.sum(axis=0, keepdims=True)
    X = (1 - gamma) * T + Z @ alpha + sigma_eps * rng.normal(size=(n, d_o))
    X = MinMaxScaler().fit_transform(X)
    Z = MinMaxScaler().fit_transform(Z)
    return X, Z, alpha

def generate_ranking(X, Z, beta=0.3, sigma_eta=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    n, d_o, d_p = X.shape[0], X.shape[1], Z.shape[1]
    w_x, w_z = np.ones(d_o), np.ones(d_p)
    gX, gZ = X @ w_x, Z @ w_z
    #S = (1-beta) * gX + beta * gZ + sigma_eta * rng.normal(size=n)
    S =  gX**(1-beta) + gZ**beta + sigma_eta * rng.normal(size=n)
    # FIX 1: La funzione ora ritorna solo R per maggiore chiarezza.
    order = S.argsort()[::-1]
    pos = np.empty_like(order)
    pos[order] = np.arange(1, n + 1)
    R = (pos - 0.5) / n
    return R

