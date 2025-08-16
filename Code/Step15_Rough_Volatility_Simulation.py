"""
Rough Volatility Simulation for SPX

This script implements rough volatility model simulation for the dissertation:
"The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies 
on Equity-Linked Variable Annuities: An Enterprise Risk Management Approach"

Author: Abdurakhmonbek Fayzullaev
Purpose: Generate SPX paths under rough volatility model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Step00_Configuration import S0, n_paths, T, N, roughvol_params, seed
from scipy.special import gamma

def simulate_rough_vol(S0, mu, T, N, n_paths, roughvol_params, seed=None):
    """
    Simulate stock price paths under rough volatility model.
    
    Args:
        S0: Initial stock price
        mu: Risk-neutral drift (r - q)
        T: Time to maturity in years
        N: Number of time steps
        n_paths: Number of simulation paths
        roughvol_params: Dictionary with rough volatility parameters
        seed: Random seed for reproducibility
    
    Returns:
        S: Stock price array of shape (n_steps+1, n_paths)
        sigma: Volatility array of shape (n_steps+1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    H = roughvol_params['H']        # Hurst parameter (H < 0.5 for roughness)
    nu = roughvol_params['nu']      # Volatility parameter
    rho = roughvol_params['rho']    # Correlation with underlying
    xi = roughvol_params['xi']      # Base volatility level
    
    # Initialize arrays
    S = np.zeros((n_paths, int(N) + 1))
    sigma = np.zeros((n_paths, int(N) + 1))
    
    # Set initial values
    S[:, 0] = S0
    sigma[:, 0] = xi
    
    # Generate fractional Brownian motion increments for volatility
    # Using the Cholesky method for correlated increments
    Z1 = np.random.normal(0, 1, (n_paths, int(N)))
    Z2 = np.random.normal(0, 1, (n_paths, int(N)))
    
    # Correlated increments for stock and volatility
    Z_S = Z1
    Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Fractional Brownian motion for volatility (simplified approach)
    # For H < 0.5, we use a moving average with power-law weights
    for t in range(int(N)):
        # Rough volatility evolution (simplified)
        if t == 0:
            sigma[:, t + 1] = xi
        else:
            # Moving average with power-law decay for roughness
            weights = np.array([(t - i + 1)**(H - 0.5) for i in range(t + 1)])
            weights = weights / np.sum(weights)
            
            # Volatility innovation
            vol_innov = nu * np.sqrt(dt) * Z_v[:, t]
            sigma[:, t + 1] = xi + np.sum(weights[:, np.newaxis] * vol_innov.reshape(1, -1), axis=0)
            sigma[:, t + 1] = np.maximum(sigma[:, t + 1], 0.01)  # Ensure positive volatility
        
        # Stock price evolution
        vol_t = sigma[:, t]
        S[:, t + 1] = S[:, t] * np.exp((mu - 0.5 * vol_t**2) * dt + vol_t * Z_S[:, t] * np.sqrt(dt))
    
    return S.T, sigma.T

def simulate_roughvol_paths(S0, T, N, n_paths, roughvol_params, seed=None):
    """
    Wrapper function to simulate rough volatility paths with default parameters.
    
    Args:
        S0: Initial stock price
        T: Time to maturity in years
        N: Number of time steps
        n_paths: Number of simulation paths
        roughvol_params: Dictionary with rough volatility parameters
        seed: Random seed for reproducibility
    
    Returns:
        S: Stock price array of shape (n_steps+1, n_paths)
    """
    mu = 0.02 - 0.015  # Default r - q
    S, sigma = simulate_rough_vol(S0, mu, T, N, n_paths, roughvol_params, seed)
    return S

if __name__ == "__main__":
    # Test rough volatility simulation
    print("Simulating SPX paths under Rough Volatility model...")
    S, sigma = simulate_rough_vol(S0, 0.005, T, N, n_paths, roughvol_params, seed)
    
    print(f"Initial price: ${S[0, 0]:.2f}")
    print(f"Average final price: ${np.mean(S[-1, :]):.2f}")
    print(f"Initial volatility: {sigma[0, 0]:.1%}")
    print(f"Average final volatility: {np.mean(sigma[-1, :]):.1%}")
    
    # Save results
    os.makedirs("Output/simulations", exist_ok=True)
    
    # Save paths
    dates = pd.date_range(start='2023-01-01', periods=N+1, freq='D')
    paths_df = pd.DataFrame(S, index=dates)
    paths_df.to_csv("Output/simulations/SPX_RoughVol_paths.csv")
    
    # Save volatility paths
    vol_df = pd.DataFrame(sigma, index=dates)
    vol_df.to_csv("Output/simulations/SPX_RoughVol_volatility.csv")
    
    # Create sample plot
    plt.figure(figsize=(15, 8))
    
    # Stock price paths
    plt.subplot(2, 1, 1)
    for i in range(min(10, n_paths)):
        plt.plot(dates, S[:, i], alpha=0.7, linewidth=0.8)
    plt.title("Sample SPX Paths under Rough Volatility Model (10 of 10,000)")
    plt.ylabel("SPX Level")
    plt.grid(True, alpha=0.3)
    
    # Volatility paths
    plt.subplot(2, 1, 2)
    for i in range(min(10, n_paths)):
        plt.plot(dates, sigma[:, i], alpha=0.7, linewidth=0.8)
    plt.title("Corresponding Rough Volatility Paths")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("Output/simulations/SPX_RoughVol_sample_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Rough Volatility simulation complete. Paths and plot saved.")
