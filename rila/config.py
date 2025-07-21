class Config:
    def __init__(self):
        self.SEED = 42
        self.S0 = 4500
        self.n_paths = 10000
        self.T = 7 
        self.N = 252 * self.T  
        self.initial_account = 1000
        self.buffer_level = 0.1  
        self.cap_level = 0.5   
        self.fee_rate = 0.01   
        self.participation_rate = 1.0 
        self.annual_reset = False 
        self.heston_params = {
            'v0': 0.0387,
            'kappa': 1.9234,
            'theta': 0.0421,
            'sigma_v': 0.292,
            'rho': -0.7
        }
        self.riskfree_file = 'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023_CLEANED.csv'
        self.dividend_file = 'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv'

config = Config()

SEED = config.SEED
S0 = config.S0
n_paths = config.n_paths
T = config.T
N = config.N
initial_account = config.initial_account
buffer_level = config.buffer_level
cap_level = config.cap_level
fee_rate = config.fee_rate
participation_rate = config.participation_rate
annual_reset = config.annual_reset
heston_params = config.heston_params
riskfree_file = config.riskfree_file
dividend_file = config.dividend_file 