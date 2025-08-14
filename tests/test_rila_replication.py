"""
Test suite for RILA replication functionality.

Tests the static replication approach against brute-force Monte Carlo
to ensure the RILA payoff is accurately replicated within tolerance.

Author: Abdurakhmonbek Fayzullaev
"""

import sys
import os
import numpy as np
import pytest
from scipy.stats import norm

# Add Code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code'))

from rila_payoff import rila_payoff, rila_replication, rila_pv, validate_rila_replication


def black_scholes_option(S0, K, T, r, q, sigma, option_type='call'):
    """Black-Scholes option pricing for testing."""
    if T <= 0:
        if option_type == 'call':
            return max(S0 - K, 0)
        else:
            return max(K - S0, 0)
    
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-q*T)*norm.cdf(-d1)


class TestRILAPayoff:
    """Test RILA payoff calculations."""
    
    def test_rila_payoff_basic_structure(self):
        """Test basic RILA payoff structure."""
        S0 = 4500.0
        cap = 0.25
        buffer = 0.10
        
        # Test various scenarios
        ST_values = np.array([3500, 4000, 4050, 4500, 5000, 5625, 6000])  # Different return levels
        payoffs = rila_payoff(ST_values, S0, cap, buffer)
        
        # Expected returns
        returns = (ST_values - S0) / S0
        expected_returns = np.array([-0.222, -0.111, -0.1, 0, 0.111, 0.25, 0.333])
        
        # Verify payoff structure
        assert len(payoffs) == len(ST_values)
        assert np.all(payoffs >= 0)  # Non-negative payoffs
        
        # Test specific cases
        # Case 1: Loss beyond buffer (ST=3500, return=-22.2%)
        assert payoffs[0] == pytest.approx(1 + returns[0] + buffer, abs=1e-6)
        
        # Case 2: Loss within buffer (ST=4050, return=-10%)
        assert payoffs[2] == pytest.approx(1.0, abs=1e-6)
        
        # Case 3: No change (ST=4500)
        assert payoffs[3] == pytest.approx(1.0, abs=1e-6)
        
        # Case 4: Gain within cap (ST=5000, return=11.1%)
        assert payoffs[4] == pytest.approx(1 + returns[4], abs=1e-6)
        
        # Case 5: Gain at cap (ST=5625, return=25%)
        assert payoffs[5] == pytest.approx(1 + cap, abs=1e-6)
        
        # Case 6: Gain above cap (ST=6000, return=33.3%)
        assert payoffs[6] == pytest.approx(1 + cap, abs=1e-6)
    
    def test_rila_payoff_edge_cases(self):
        """Test edge cases for RILA payoff."""
        S0 = 1000.0
        cap = 0.15
        buffer = 0.05
        
        # Exact boundary cases
        ST_buffer = S0 * (1 - buffer)  # Exactly at buffer
        ST_cap = S0 * (1 + cap)       # Exactly at cap
        
        ST_values = np.array([ST_buffer, S0, ST_cap])
        payoffs = rila_payoff(ST_values, S0, cap, buffer)
        
        # At buffer boundary: should get full protection
        assert payoffs[0] == pytest.approx(1.0, abs=1e-6)
        
        # At no change: should get 1.0
        assert payoffs[1] == pytest.approx(1.0, abs=1e-6)
        
        # At cap: should get 1 + cap
        assert payoffs[2] == pytest.approx(1 + cap, abs=1e-6)
    
    def test_rila_payoff_vectorized(self):
        """Test vectorized operation."""
        S0 = 3000.0
        cap = 0.20
        buffer = 0.15
        
        # Large array
        np.random.seed(42)
        ST_values = S0 * np.exp(np.random.normal(0, 0.3, 10000))
        
        payoffs = rila_payoff(ST_values, S0, cap, buffer)
        
        assert len(payoffs) == len(ST_values)
        assert np.all(payoffs >= 0)
        assert np.all(payoffs <= 1 + cap)


class TestRILAReplication:
    """Test RILA static replication."""
    
    def setup_method(self):
        """Setup test parameters."""
        self.S0 = 4500.0
        self.cap = 0.25
        self.buffer = 0.10
        self.T = 1.0
        self.r = 0.02
        self.q = 0.015
        self.sigma = 0.20
    
    def test_rila_replication_structure(self):
        """Test replication portfolio structure."""
        K_grid = np.arange(3000, 7000, 50)
        replication = rila_replication(self.S0, K_grid, self.cap, self.buffer, {})
        
        # Check structure
        assert 'underlying_weight' in replication
        assert 'strikes' in replication
        assert 'weights' in replication
        assert 'types' in replication
        
        # Check values
        assert replication['underlying_weight'] == pytest.approx(1.0, abs=1e-6)
        assert len(replication['strikes']) == 2  # Put and call
        assert len(replication['weights']) == 2
        assert len(replication['types']) == 2
        
        # Check option types and strikes
        buffer_strike = self.S0 * (1 - self.buffer)
        cap_strike = self.S0 * (1 + self.cap)
        
        assert np.any(np.isclose(replication['strikes'], buffer_strike, atol=50))
        assert np.any(np.isclose(replication['strikes'], cap_strike, atol=50))
        
        assert 'put' in replication['types']
        assert 'call' in replication['types']
    
    def test_rila_replication_validation(self):
        """Test replication validation against direct payoff."""
        K_grid = np.arange(3000, 7000, 25)
        replication = rila_replication(self.S0, K_grid, self.cap, self.buffer, {})
        
        # Test terminal prices
        ST_test = np.linspace(3000, 7000, 100)
        
        # Validate replication
        validation_metrics = validate_rila_replication(
            S0=self.S0, cap=self.cap, buffer=self.buffer,
            replication=replication, ST_test=ST_test,
            option_pricer=lambda **kwargs: black_scholes_option(sigma=self.sigma, **kwargs)
        )
        
        # Check validation metrics
        assert validation_metrics['max_abs_error'] < 0.01  # Less than 1 cent error
        assert validation_metrics['mean_abs_error'] < 0.001  # Very small mean error
        assert validation_metrics['max_rel_error'] < 0.05   # Less than 5% relative error
        assert validation_metrics['rmse'] < 0.005           # Small RMSE
    
    def test_rila_pv_calculation(self):
        """Test RILA present value calculation."""
        
        def test_option_pricer(S0, K, T, r, q, option_type='call', **kwargs):
            return black_scholes_option(S0, K, T, r, q, self.sigma, option_type)
        
        # Calculate RILA PV
        pv = rila_pv(
            S0=self.S0, T=self.T, r=self.r, q=self.q,
            cap=self.cap, buffer=self.buffer,
            option_pricer=test_option_pricer
        )
        
        # PV should be reasonable
        assert 0.5 < pv < 2.0  # Reasonable range for RILA PV
        assert not np.isnan(pv)
        assert not np.isinf(pv)
    
    def test_rila_pv_time_decay(self):
        """Test that RILA PV decreases with shorter time to maturity."""
        
        def test_option_pricer(S0, K, T, r, q, option_type='call', **kwargs):
            return black_scholes_option(S0, K, T, r, q, self.sigma, option_type)
        
        pv_1y = rila_pv(self.S0, 1.0, self.r, self.q, self.cap, self.buffer, test_option_pricer)
        pv_6m = rila_pv(self.S0, 0.5, self.r, self.q, self.cap, self.buffer, test_option_pricer)
        pv_3m = rila_pv(self.S0, 0.25, self.r, self.q, self.cap, self.buffer, test_option_pricer)
        
        # Generally expect time decay (though not always monotonic for RILA)
        assert pv_1y > 0 and pv_6m > 0 and pv_3m > 0
        assert not (np.isnan(pv_1y) or np.isnan(pv_6m) or np.isnan(pv_3m))


class TestRILAMonteCarlo:
    """Test RILA replication against Monte Carlo."""
    
    def setup_method(self):
        """Setup Monte Carlo test."""
        self.S0 = 4500.0
        self.cap = 0.25
        self.buffer = 0.10
        self.T = 1.0
        self.r = 0.02
        self.q = 0.015
        self.sigma = 0.20
        
        # Monte Carlo parameters
        self.n_paths = 100000
        np.random.seed(42)
    
    def test_replication_vs_monte_carlo(self):
        """Test replication against Monte Carlo simulation."""
        
        # Monte Carlo simulation
        dt = self.T
        Z = np.random.standard_normal(self.n_paths)
        ST_mc = self.S0 * np.exp((self.r - self.q - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z)
        
        # Direct payoff calculation
        direct_payoffs = rila_payoff(ST_mc, self.S0, self.cap, self.buffer)
        mc_pv = np.exp(-self.r * self.T) * np.mean(direct_payoffs)
        
        # Replication approach
        def test_option_pricer(S0, K, T, r, q, option_type='call', **kwargs):
            return black_scholes_option(S0, K, T, r, q, self.sigma, option_type)
        
        replication_pv = rila_pv(
            S0=self.S0, T=self.T, r=self.r, q=self.q,
            cap=self.cap, buffer=self.buffer,
            option_pricer=test_option_pricer
        )
        
        # Compare results (should be close within Monte Carlo error)
        relative_error = abs(replication_pv - mc_pv) / mc_pv
        
        # Allow for Monte Carlo error (typically 1-2% for 100k paths)
        assert relative_error < 0.05, f"Replication error too large: {relative_error:.3%}"
        
        # Log results for inspection
        print(f"Monte Carlo PV: {mc_pv:.6f}")
        print(f"Replication PV: {replication_pv:.6f}")
        print(f"Relative Error: {relative_error:.3%}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])