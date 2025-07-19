"""
Simulation models for RILA analysis: GBM, Heston, and Rough Volatility.
All functions are vectorized and reproducible.
"""
import numpy as np
import pandas as pd

def simulate_gbm(S0, mu, sigma, T, N, n_paths, seed=42):
    """
    Simulate Geometric Brownian Motion (GBM) paths.
    Args:
        S0: initial value
        mu: drift
        sigma: volatility
        T: time horizon (years)
        N: number of steps
        n_paths: number of paths
        seed: random seed
    Returns:
        np.ndarray of shape (N+1, n_paths)
    """
    np.random.seed(seed)
    dt = T / N
    S = np.zeros((N + 1, n_paths))
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.normal(size=n_paths)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S

def simulate_heston(S0, v0, mu, kappa, theta, sigma_v, rho, T, N, n_paths, seed=42):
    """
    Simulate Heston model paths.
    Args:
        S0: initial value
        v0: initial variance
        mu: drift
        kappa, theta, sigma_v, rho: Heston parameters
        T: time horizon (years)
        N: number of steps
        n_paths: number of paths
        seed: random seed
    Returns:
        np.ndarray of shape (N+1, n_paths)
    """
    np.random.seed(seed)
    dt = T / N
    S = np.zeros((N + 1, n_paths))
    V = np.zeros((N + 1, n_paths))
    S[0] = S0
    V[0] = v0
    Z1 = np.random.normal(size=(N, n_paths))
    Z2 = np.random.normal(size=(N, n_paths))
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    for t in range(1, N + 1):
        V[t] = np.abs(V[t-1] + kappa * (theta - V[t-1]) * dt + sigma_v * np.sqrt(V[t-1]) * np.sqrt(dt) * W2[t-1])
        S[t] = S[t-1] * np.exp((mu - 0.5 * V[t-1]) * dt + np.sqrt(V[t-1]) * np.sqrt(dt) * W1[t-1])
    return S

def simulate_rough_vol(S0, mu, xi0, eta, H, T, N, n_paths, seed=42):
    """
    Vectorized simulation of rough volatility model paths (lognormal fractional volatility).
    Args:
        S0: initial value
        mu: drift
        xi0: initial variance
        eta: vol of vol
        H: Hurst parameter
        T: time horizon (years)
        N: number of steps
        n_paths: number of paths
        seed: random seed
    Returns:
        np.ndarray of shape (N+1, n_paths)
    """
    np.random.seed(seed)
    dt = T / N
    S = np.zeros((N + 1, n_paths))
    S[0] = S0
    # Vectorized fBM increments for all paths
    W_H = np.cumsum(np.random.normal(size=(N+1, n_paths)), axis=0) * dt**H
    t_grid = np.arange(N+1) * dt
    v_t = xi0 * np.exp(eta * (W_H - 0.5 * eta**2 * t_grid[:, None]**(2*H)))
    dW = np.random.normal(size=(N, n_paths)) * np.sqrt(dt)
    drift_adj = (mu - 0.5 * v_t[:-1]) * dt
    diffusion = np.sqrt(v_t[:-1]) * dW
    log_returns = drift_adj + diffusion
    S[1:] = S0 * np.exp(np.cumsum(log_returns, axis=0))
    return S 