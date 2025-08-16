"""
Heston Stochastic Volatility Simulation for SPX

This script implements Heston model simulation for the dissertation:
"The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies 
on Equity-Linked Variable Annuities: An Enterprise Risk Management Approach"

Author: Abdurakhmonbek Fayzullaev
Purpose: Generate SPX paths under Heston stochastic volatility model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def simulate_heston(S0, mu, T, N, n_paths, heston_params, seed=None):
    """
    Simulate stock price paths under Heston stochastic volatility model.
    
    Args:
        S0: Initial stock price
        mu: Risk-neutral drift (r - q)
        T: Time to maturity in years
        N: Number of time steps
        n_paths: Number of simulation paths
        heston_params: Dictionary with Heston parameters
        seed: Random seed for reproducibility
    
    Returns:
        S: Stock price array of shape (n_steps+1, n_paths)
        v: Variance array of shape (n_steps+1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    
    # Extract Heston parameters
    v0 = heston_params['v0']      # Initial variance
    kappa = heston_params['kappa'] # Mean reversion speed
    theta = heston_params['theta'] # Long-term variance
    sigma_v = heston_params['sigma_v'] # Volatility of volatility
    rho = heston_params['rho']    # Stock-volatility correlation
    
    # Initialize arrays
    S = np.zeros((n_paths, int(N) + 1))
    v = np.zeros((n_paths, int(N) + 1))
    
    # Set initial values
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Generate correlated random numbers
    Z1 = np.random.normal(0, 1, (n_paths, int(N)))
    Z2 = np.random.normal(0, 1, (n_paths, int(N)))
    Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Euler-Maruyama scheme for Heston model
    for t in range(int(N)):
        # Variance process (with Feller condition check)
        v_old = v[:, t]
        v_new = v_old + kappa * (theta - v_old) * dt + sigma_v * np.sqrt(np.maximum(v_old, 0)) * Z_v[:, t] * np.sqrt(dt)
        v[:, t + 1] = np.maximum(v_new, 0)  # Ensure variance is non-negative
        
        # Stock price process
        vol = np.sqrt(np.maximum(v[:, t], 1e-6))  # Avoid division by zero
        S[:, t + 1] = S[:, t] * np.exp((mu - 0.5 * vol**2) * dt + vol * Z1[:, t] * np.sqrt(dt))
    
    return S.T, v.T

def simulate_heston_paths(S0, T, N, n_paths, heston_params, seed=None):
    """
    Wrapper function to simulate Heston paths with default parameters.
    
    Args:
        S0: Initial stock price
        T: Time to maturity in years
        N: Number of time steps
        n_paths: Number of simulation paths
        heston_params: Dictionary with Heston parameters
        seed: Random seed for reproducibility
    
    Returns:
        S: Stock price array of shape (n_steps+1, n_paths)
    """
    mu = 0.02 - 0.015  # Default r - q
    S, v = simulate_heston(S0, mu, T, N, n_paths, heston_params, seed)
    return S

if __name__ == "__main__":
    # Test Heston simulation
    import sys
    sys.path.append('.')
    from Step00_Configuration import S0, n_paths, T, N, heston_params, seed
    
    print("Simulating SPX paths under Heston model...")
    S, v = simulate_heston(S0, 0.005, T, N, n_paths, heston_params, seed)
    
    print(f"Initial price: ${S[0, 0]:.2f}")
    print(f"Average final price: ${np.mean(S[-1, :]):.2f}")
    print(f"Initial volatility: {np.sqrt(v[0, 0]):.1%}")
    print(f"Average final volatility: {np.mean(np.sqrt(v[-1, :])):.1%}")
    
    # Save results
    os.makedirs("Output/simulations", exist_ok=True)
    
    # Save paths
    dates = pd.date_range(start='2023-01-01', periods=int(N)+1, freq='D')
    paths_df = pd.DataFrame(S, index=dates)
    paths_df.to_csv("Output/simulations/SPX_Heston_paths.csv")
    
    # Save volatility paths
    vol_df = pd.DataFrame(np.sqrt(v), index=dates)
    vol_df.to_csv("Output/simulations/SPX_Heston_volatility.csv")
    
    # Create sample plot
    plt.figure(figsize=(15, 8))
    
    # Stock price paths
    plt.subplot(2, 1, 1)
    for i in range(min(10, n_paths)):
        plt.plot(dates, S[:, i], alpha=0.7, linewidth=0.8)
    plt.title("Sample SPX Paths under Heston Model (10 of 10,000)")
    plt.ylabel("SPX Level")
    plt.grid(True, alpha=0.3)
    
    # Volatility paths
    plt.subplot(2, 1, 2)
    for i in range(min(10, n_paths)):
        plt.plot(dates, np.sqrt(v[:, i]), alpha=0.7, linewidth=0.8)
    plt.title("Corresponding Volatility Paths")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("Output/simulations/SPX_Heston_sample_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Heston simulation complete. Paths and plot saved.")
