import numpy as np
from rila.models import simulate_gbm, simulate_heston, simulate_rough_vol

def test_simulate_gbm_shape():
    S = simulate_gbm(100, 0.01, 0.2, 1, 10, 5, seed=1)
    assert S.shape == (11, 5)

def test_simulate_heston_shape():
    S = simulate_heston(100, 0.04, 0.01, 2.0, 0.04, 0.3, -0.7, 1, 10, 5, seed=1)
    assert S.shape == (11, 5)

def test_simulate_rough_vol_shape():
    S = simulate_rough_vol(100, 0.01, 0.04, 1.5, 0.1, 1, 10, 5, seed=1)
    assert S.shape == (11, 5)

def test_simulate_gbm_nonnegative():
    S = simulate_gbm(100, 0.01, 0.2, 1, 10, 5, seed=1)
    assert np.all(S > 0) 