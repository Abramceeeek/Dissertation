S0 = 4500
n_paths = 10000
T = 7  
N = 252 * T 
initial_account = 1000
buffer_level = 0.1 
cap_level = 0.5   
fee_rate = 0.01 
heston_params = {
    'v0': 0.04,
    'kappa': 2.0,
    'theta': 0.04,
    'sigma_v': 0.3,
    'rho': -0.7
}
riskfree_file = 'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023_CLEANED.csv'
dividend_file = 'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv' 