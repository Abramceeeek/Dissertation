"""
Variable Annuity Configuration File

This file contains all the key parameters used in the dissertation:
"The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies 
on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk Management Approach"

Author: Abdurakhmonbek Fayzullaev
Purpose: Centralized configuration for Variable Annuity modeling and risk management
"""

# Market Parameters
S0 = 4500  # Initial S&P 500 index level (reflects 2018 market conditions)
n_paths = 10000  # Number of Monte Carlo simulation paths for robust statistical analysis
T = 7  # Time horizon in years (typical variable annuity contract length)
N = 252 * T  # Total number of time steps (daily observations over contract period)

# Variable Annuity Product Parameters
initial_account = 1000  # Initial account value (policyholder's premium payment)
guarantee_rate = 0.03  # Minimum guaranteed return rate (3% annual guarantee)
management_fee = 0.015  # Annual management fee charged by insurance company
guarantee_fee = 0.005  # Additional fee for guarantee provision
mortality_charge = 0.001  # Annual mortality and expense charge

# Risk Management Parameters  
hedge_threshold = 0.10  # Delta threshold for hedge rebalancing (10% change triggers rebalancing)
transaction_cost = 0.001  # Proportional transaction cost for hedging trades (10 basis points)

# Heston Stochastic Volatility Model Parameters
# These parameters are calibrated to S&P 500 market data (2018-2023)
heston_params = {
    'v0': 0.04,      # Initial variance (4% volatility level at start)
    'kappa': 2.0,    # Mean reversion speed (how quickly volatility returns to long-term mean)
    'theta': 0.04,   # Long-term variance level (target volatility of 20%)
    'sigma_v': 0.3,  # Volatility of volatility (vol-of-vol parameter)
    'rho': -0.7      # Correlation between stock and volatility (negative = leverage effect)
}

# Rough Volatility Model Parameters (for comparison with Heston model)
rough_vol_params = {
    'H': 0.1,        # Hurst parameter (H < 0.5 indicates rough/anti-persistent behavior)
    'nu': 0.3,       # Volatility parameter 
    'rho': -0.7      # Correlation with underlying asset
}

# Data File Locations (relative paths from project root)
riskfree_file = 'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023_CLEANED.csv'
dividend_file = 'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv'
spx_options_file = 'Data/SPX Option Chain/SPX_Options_Data_2018_to_2023_MERGED.csv'

# Capital Requirements Parameters (Solvency II inspired)
confidence_level = 0.995  # 99.5% confidence level for capital requirements
risk_measure = 'VaR'      # Risk measure: 'VaR' or 'ES' (Expected Shortfall)
time_horizon_days = 252   # One-year time horizon for capital calculation

# Output Settings
save_plots = True         # Whether to save visualization plots
output_precision = 4      # Decimal places for numerical results
random_seed = 42          # Seed for reproducible Monte Carlo simulations 