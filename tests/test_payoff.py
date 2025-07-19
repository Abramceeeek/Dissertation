import numpy as np
from rila.payoff import apply_rila_payoff

def test_rila_payoff_positive_within_cap():
    returns = np.array([0.05, 0.10, 0.20])
    buffer = 0.1
    cap = 0.15
    credited = apply_rila_payoff(returns, buffer, cap)
    expected = np.array([0.05, 0.10, 0.15])
    assert np.allclose(credited, expected)

def test_rila_payoff_negative_within_buffer():
    returns = np.array([-0.05, -0.09])
    buffer = 0.1
    cap = 0.15
    credited = apply_rila_payoff(returns, buffer, cap)
    expected = np.array([0.0, 0.0])
    assert np.allclose(credited, expected)

def test_rila_payoff_negative_beyond_buffer():
    returns = np.array([-0.15, -0.20])
    buffer = 0.1
    cap = 0.15
    credited = apply_rila_payoff(returns, buffer, cap)
    expected = np.array([-0.05, -0.10])
    assert np.allclose(credited, expected)

def test_rila_payoff_at_cap():
    returns = np.array([0.15, 0.20, 0.50])
    buffer = 0.1
    cap = 0.15
    credited = apply_rila_payoff(returns, buffer, cap)
    expected = np.array([0.15, 0.15, 0.15])
    assert np.allclose(credited, expected) 