"""
Unified Configuration for GMAB Variable Annuity Analysis

This file contains all unified parameters used in the dissertation:
"The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies 
on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk Management Approach"

Author: Abdurakhmonbek Fayzullaev
Purpose: Centralized configuration for GMAB modeling and risk management
"""

import numpy as np

# =====================================
# Market and Simulation Parameters
# =====================================

S0 = 4500  # Initial S&P 500 index level (unified across all models)
n_paths = 10000  # Number of Monte Carlo simulation paths
T = 7.0  # Contract maturity in years (unified to exactly 7 years)
N = 252 * T  # Total number of daily time steps
dt_years = T / N  # Time step size in years

# Random seed for reproducibility
seed = 42
use_common_random_numbers = True  # Use same random numbers across models for comparison

# =====================================
# GMAB Product Parameters
# =====================================

# Core GMAB parameters
g_annual = 0.03  # Guaranteed minimum accumulation rate (3% p.a.)
initial_premium = 1.0  # Normalized initial premium
fee_annual = 0.015  # Annual management fee (1.5%)

# Transaction costs and rebalancing
trans_cost_bps = 10.0  # Transaction cost in basis points (10 bps)
rebalance_freq = 7  # Rebalancing frequency in days (weekly)

# Additional fees
guarantee_fee = 0.005  # Additional guarantee fee
mortality_charge = 0.001  # Mortality and expense charge

# =====================================
# Market Data Sources (Cleaned)
# =====================================

# Use cleaned risk-free rate data with maturity_days column
riskfree_file = 'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023_CLEANED.csv'
dividend_file = 'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv'

# Default risk-free rate and dividend yield (fallback values)
r_default = 0.02  # 2% risk-free rate
q_default = 0.015  # 1.5% dividend yield

# =====================================
# Model Parameters
# =====================================

# GBM Parameters
gbm_params = {
    'mu': r_default - q_default,  # Risk-neutral drift
    'sigma': 0.2  # Constant volatility (20%)
}

# Heston Stochastic Volatility Parameters (calibrated)
heston_params = {
    'v0': 0.04,      # Initial variance (20% initial volatility)
    'kappa': 2.0,    # Mean reversion speed
    'theta': 0.04,   # Long-term variance (20% long-term volatility)
    'sigma_v': 0.3,  # Volatility of volatility
    'rho': -0.7      # Stock-volatility correlation (leverage effect)
}

# Rough Volatility Parameters
roughvol_params = {
    'H': 0.1,        # Hurst parameter (rough behavior, H < 0.5)
    'nu': 0.3,       # Volatility parameter
    'rho': -0.7,     # Correlation with underlying
    'xi': 0.2        # Base volatility level
}

# =====================================
# Hedging and Risk Management
# =====================================

# Delta hedging parameters
hedge_threshold = 0.05  # Delta threshold for rebalancing (5% change)
lambda_scale = 1.0  # Scaling factor for option values

# Risk metrics calculation
var_confidence = 0.995  # 99.5% VaR confidence level
cte_confidence = 0.995  # 99.5% CTE confidence level

# =====================================
# Output and Results Configuration
# =====================================

# Results directories
results_dir = "results"
tables_dir = "tables"
figures_dir = "figures"

# Output file naming
results_gbm_file = f"{results_dir}/results_gbm_gmab.csv"
results_heston_file = f"{results_dir}/results_heston_gmab.csv"
results_roughvol_file = f"{results_dir}/results_roughvol_gmab.csv"

# LaTeX table files
table_gbm_file = f"{tables_dir}/table_gbm.tex"
table_heston_file = f"{tables_dir}/table_heston.tex"
table_roughvol_file = f"{tables_dir}/table_roughvol.tex"

# Figure output settings
figure_dpi = 300
figure_format = 'png'

# =====================================
# Utility Functions
# =====================================

def get_rebalance_times(n_steps: int, freq_days: int) -> np.ndarray:
    """Get rebalancing time indices."""
    return np.arange(0, n_steps + 1, freq_days)

def calculate_var_cte(pnl_data: np.ndarray, confidence: float = 0.995) -> tuple:
    """Calculate VaR and CTE from P&L data."""
    var = np.percentile(pnl_data, (1 - confidence) * 100)
    cte = np.mean(pnl_data[pnl_data <= var])
    return var, cte

def format_risk_metrics(pnl_data: np.ndarray) -> dict:
    """Format risk metrics for output."""
    var, cte = calculate_var_cte(pnl_data, var_confidence)
    
    return {
        'mean_pnl': np.mean(pnl_data),
        'std_pnl': np.std(pnl_data),
        'var_99_5': var,
        'cte_99_5': cte,
        'min_pnl': np.min(pnl_data),
        'max_pnl': np.max(pnl_data)
    }

# =====================================
# Validation and Constraints
# =====================================

def validate_heston_params(params: dict) -> bool:
    """Validate Heston parameters satisfy Feller condition."""
    return 2 * params['kappa'] * params['theta'] > params['sigma_v'] ** 2

def validate_config():
    """Validate configuration consistency."""
    assert T > 0, "Contract maturity must be positive"
    assert g_annual > 0, "Guarantee rate must be positive"
    assert fee_annual >= 0, "Management fee must be non-negative"
    assert trans_cost_bps >= 0, "Transaction costs must be non-negative"
    assert 0 < var_confidence < 1, "VaR confidence must be between 0 and 1"
    
    if not validate_heston_params(heston_params):
        print(f"WARNING: Heston parameters may violate Feller condition")
        print(f"2*kappa*theta = {2 * heston_params['kappa'] * heston_params['theta']:.4f}")
        print(f"sigma_v^2 = {heston_params['sigma_v']**2:.4f}")
    
    print("Configuration validation completed successfully")

if __name__ == "__main__":
    # Validate configuration when run as script
    validate_config()
    
    # Print key parameters
    print("=" * 50)
    print("GMAB Configuration Summary")
    print("=" * 50)
    print(f"Initial Stock Price (S0): ${S0}")
    print(f"Contract Maturity (T): {T} years")
    print(f"Guarantee Rate: {g_annual:.1%} p.a.")
    print(f"Management Fee: {fee_annual:.1%} p.a.")
    print(f"Number of Paths: {n_paths:,}")
    print(f"Simulation Seed: {seed}")
    print(f"Rebalancing Frequency: Every {rebalance_freq} days")
    print(f"Transaction Costs: {trans_cost_bps} bps")
    print("=" * 50)