import numpy as np
# from rila.models import simulate_gbm, simulate_heston, simulate_rough_vol  # Module not found
from Step13_GBM_Simulation import simulate_gbm
from Step14_Heston_Simulation import simulate_heston  
from Step15_Rough_Volatility_Simulation import simulate_rough_vol

def test_simulate_gbm_shape():
    S = simulate_gbm(S0=100, mu=0.01, sigma=0.2, T=1, N=10, n_paths=5, seed=1)
    assert S.shape == (11, 5)

def test_simulate_heston_shape():
    heston_params = {'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.7}
    S, V = simulate_heston(S0=100, mu=0.01, T=1, N=10, n_paths=5, heston_params=heston_params, seed=1)
    assert S.shape == (11, 5)

def test_simulate_rough_vol_shape():
    roughvol_params = {'H': 0.1, 'xi': 0.04, 'nu': 1.5, 'rho': -0.7}
    S, sigma = simulate_rough_vol(S0=100, mu=0.01, T=1, N=10, n_paths=5, roughvol_params=roughvol_params, seed=1)
    assert S.shape == (11, 5)

def test_simulate_gbm_nonnegative():
    S = simulate_gbm(S0=100, mu=0.01, sigma=0.2, T=1, N=10, n_paths=5, seed=1)
    assert np.all(S > 0) 