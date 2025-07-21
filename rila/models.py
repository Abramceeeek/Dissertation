import numpy as np
import pandas as pd
from rila.config import SEED
from concurrent.futures import ProcessPoolExecutor

def simulate_gbm(S0, mu, sigma, T, N, n_paths, seed=SEED):
    np.random.seed(seed)
    dt = T / N
    S = np.zeros((N + 1, n_paths))
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.normal(size=n_paths)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S

def _simulate_heston_chunk(chunk_indices, S0, v0, mu, kappa, theta, sigma_v, rho, T, N, seed, feller_violated):
    np.random.seed(seed + chunk_indices[0])
    n_chunk = len(chunk_indices)
    dt = T / N
    S = np.zeros((N + 1, n_chunk))
    V = np.zeros((N + 1, n_chunk))
    S[0] = S0
    V[0] = v0
    Z1 = np.random.normal(size=(N, n_chunk))
    Z2 = np.random.normal(size=(N, n_chunk))
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    for t in range(1, N + 1):
        v_new = V[t-1] + kappa * (theta - V[t-1]) * dt + sigma_v * np.sqrt(V[t-1]) * np.sqrt(dt) * W2[t-1]
        if feller_violated:
            v_new = np.maximum(v_new, 0)
        V[t] = v_new
        S[t] = S[t-1] * np.exp((mu - 0.5 * V[t-1]) * dt + np.sqrt(V[t-1]) * np.sqrt(dt) * W1[t-1])
    return S

def simulate_heston_parallel(S0, v0, mu, kappa, theta, sigma_v, rho, T, N, n_paths, n_workers=4, seed=SEED):
    feller_violated = 2 * kappa * theta <= sigma_v**2
    indices = np.array_split(np.arange(n_paths), n_workers)
    args = [(chunk, S0, v0, mu, kappa, theta, sigma_v, rho, T, N, seed, feller_violated) for chunk in indices]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(lambda p: _simulate_heston_chunk(*p), args))
    return np.hstack(results)

def simulate_heston(S0, v0, mu, kappa, theta, sigma_v, rho, T, N, n_paths, seed=SEED):
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
    feller_violated = 2 * kappa * theta <= sigma_v**2
    for t in range(1, N + 1):
        v_new = V[t-1] + kappa * (theta - V[t-1]) * dt + sigma_v * np.sqrt(V[t-1]) * np.sqrt(dt) * W2[t-1]
        if feller_violated:
            v_new = np.maximum(v_new, 0)
        V[t] = v_new
        S[t] = S[t-1] * np.exp((mu - 0.5 * V[t-1]) * dt + np.sqrt(V[t-1]) * np.sqrt(dt) * W1[t-1])
    return S

def simulate_rough_vol(S0, mu, xi0, eta, H, T, N, n_paths, seed=SEED):
    np.random.seed(seed)
    dt = T / N
    S = np.zeros((N + 1, n_paths))
    S[0] = S0
    W_H = np.cumsum(np.random.normal(size=(N+1, n_paths)), axis=0) * dt**H
    t_grid = np.arange(N+1) * dt
    v_t = xi0 * np.exp(eta * (W_H - 0.5 * eta**2 * t_grid[:, None]**(2*H)))
    dW = np.random.normal(size=(N, n_paths)) * np.sqrt(dt)
    drift_adj = (mu - 0.5 * v_t[:-1]) * dt
    diffusion = np.sqrt(v_t[:-1]) * dW
    log_returns = drift_adj + diffusion
    S[1:] = S0 * np.exp(np.cumsum(log_returns, axis=0))
    return S 