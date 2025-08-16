"""
Test suite for SCR (Solvency Capital Requirement) metrics.

Tests VaR and CTE calculation functions to ensure they return correct
quantiles and statistics on synthetic data.

Author: Abdurakhmonbek Fayzullaev
"""

import sys
import os
import numpy as np
import pytest
from scipy import stats

# Add Code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code'))

from Step26_Solvency_II_SCR_Calculation import calculate_var_cte, compute_one_year_scr


class TestVarCteCalculations:
    """Test VaR and CTE calculation functions."""
    
    def test_var_cte_normal_distribution(self):
        """Test VaR and CTE on normal distribution with known values."""
        np.random.seed(42)
        
        # Generate normal distribution (mean=0, std=1)
        n_samples = 100000
        losses = np.random.normal(0, 1, n_samples)
        
        # Calculate VaR and CTE
        results = calculate_var_cte(losses, confidence_levels=[0.95, 0.99, 0.995])
        
        # Theoretical values for standard normal
        theoretical_var_95 = stats.norm.ppf(0.95)
        theoretical_var_99 = stats.norm.ppf(0.99)
        theoretical_var_995 = stats.norm.ppf(0.995)
        
        # Theoretical CTE (Expected Shortfall) for standard normal
        theoretical_cte_95 = stats.norm.pdf(theoretical_var_95) / (1 - 0.95)
        theoretical_cte_99 = stats.norm.pdf(theoretical_var_99) / (1 - 0.99)
        theoretical_cte_995 = stats.norm.pdf(theoretical_var_995) / (1 - 0.995)
        
        # Test VaR values (should be close to theoretical)
        assert abs(results['VaR_95.0'] - theoretical_var_95) < 0.02
        assert abs(results['VaR_99.0'] - theoretical_var_99) < 0.02
        assert abs(results['VaR_99.5'] - theoretical_var_995) < 0.02
        
        # Test CTE values (should be close to theoretical)
        assert abs(results['CTE_95.0'] - theoretical_cte_95) < 0.02
        assert abs(results['CTE_99.0'] - theoretical_cte_99) < 0.02  
        assert abs(results['CTE_99.5'] - theoretical_cte_995) < 0.02
        
        # Test basic properties
        assert results['VaR_99.5'] > results['VaR_99.0']
        assert results['VaR_99.0'] > results['VaR_95.0']
        assert results['CTE_99.5'] > results['VaR_99.5']
        
        # Test summary statistics
        assert abs(results['mean']) < 0.02  # Should be close to 0
        assert abs(results['std'] - 1.0) < 0.02  # Should be close to 1
    
    def test_var_cte_uniform_distribution(self):
        """Test VaR and CTE on uniform distribution."""
        np.random.seed(123)
        
        # Generate uniform distribution [0, 1]
        n_samples = 50000
        losses = np.random.uniform(0, 1, n_samples)
        
        results = calculate_var_cte(losses, confidence_levels=[0.9, 0.95, 0.99])
        
        # For uniform [0,1], 90th percentile should be 0.9, 95th should be 0.95, etc.
        assert abs(results['VaR_90.0'] - 0.9) < 0.01
        assert abs(results['VaR_95.0'] - 0.95) < 0.01
        assert abs(results['VaR_99.0'] - 0.99) < 0.01
        
        # CTE should be higher than VaR
        assert results['CTE_90.0'] > results['VaR_90.0']
        assert results['CTE_95.0'] > results['VaR_95.0']
        assert results['CTE_99.0'] > results['VaR_99.0']
        
        # For uniform distribution, mean should be 0.5
        assert abs(results['mean'] - 0.5) < 0.01
    
    def test_var_cte_extreme_values(self):
        """Test VaR and CTE with extreme confidence levels."""
        np.random.seed(456)
        
        # Generate data with some extreme values
        losses = np.concatenate([
            np.random.normal(0, 1, 9900),  # 99% normal data
            np.random.normal(5, 1, 100)    # 1% extreme values
        ])
        
        results = calculate_var_cte(losses, confidence_levels=[0.99, 0.999])
        
        # Basic checks
        assert results['VaR_99.0'] > 2  # Should capture some extreme values
        assert results['VaR_99.9'] > results['VaR_99.0']
        assert results['CTE_99.9'] > results['VaR_99.9']
        assert results['CTE_99.0'] > results['VaR_99.0']
    
    def test_var_cte_constant_values(self):
        """Test VaR and CTE with constant values."""
        constant_value = 100.0
        losses = np.full(1000, constant_value)
        
        results = calculate_var_cte(losses, confidence_levels=[0.95, 0.99])
        
        # All quantiles should equal the constant value
        assert results['VaR_95.0'] == constant_value
        assert results['VaR_99.0'] == constant_value
        assert results['CTE_95.0'] == constant_value
        assert results['CTE_99.0'] == constant_value
        
        # Statistics
        assert results['mean'] == constant_value
        assert results['std'] == 0.0
        assert results['min'] == constant_value
        assert results['max'] == constant_value
    
    def test_var_cte_single_extreme_value(self):
        """Test behavior with single extreme outlier."""
        # Most values near zero, one extreme value
        losses = np.concatenate([
            np.zeros(999),
            np.array([1000.0])  # Single extreme value
        ])
        
        results = calculate_var_cte(losses, confidence_levels=[0.99, 0.999])
        
        # 99th percentile should still be 0 (only 0.1% extreme)
        assert results['VaR_99.0'] == 0.0
        
        # But 99.9th percentile should capture the extreme value
        assert results['VaR_99.9'] == 1000.0
        
        # CTE should equal extreme value when it's captured
        assert results['CTE_99.9'] == 1000.0


class TestSyntheticSCRData:
    """Test SCR calculation with controlled synthetic data."""
    
    def setup_method(self):
        """Setup synthetic test data."""
        self.n_paths = 10000
        self.n_steps = 252  # Daily steps for 1 year
        self.S0 = 4500.0
        self.T = 5.0  # 5-year product
        np.random.seed(789)
    
    def create_simple_option_pricer(self, vol=0.2):
        """Create simple Black-Scholes option pricer for testing."""
        from scipy.stats import norm
        
        def price_option(S0, K, T, r, q, option_type='call', **kwargs):
            if T <= 0:
                if option_type == 'call':
                    return max(S0 - K, 0)
                else:
                    return max(K - S0, 0)
            
            d1 = (np.log(S0/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
            d2 = d1 - vol*np.sqrt(T)
            
            if option_type == 'call':
                return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-q*T)*norm.cdf(-d1)
        
        return price_option
    
    def generate_gbm_paths(self):
        """Generate simple GBM paths for testing."""
        dt = 1.0 / self.n_steps
        t_grid = np.linspace(0, 1, self.n_steps + 1)
        
        # GBM parameters
        mu = 0.05  # Drift
        sigma = 0.2  # Volatility
        
        # Generate paths
        dW = np.random.normal(0, np.sqrt(dt), (self.n_paths, self.n_steps))
        S_paths = np.zeros((self.n_paths, self.n_steps + 1))
        S_paths[:, 0] = self.S0
        
        for i in range(self.n_steps):
            S_paths[:, i+1] = S_paths[:, i] * np.exp(
                (mu - 0.5*sigma**2)*dt + sigma*dW[:, i]
            )
        
        return S_paths, t_grid
    
    def test_scr_calculation_structure(self):
        """Test that SCR calculation returns proper structure."""
        S_paths, t_grid = self.generate_gbm_paths()
        
        # RILA parameters
        rila_params = {
            'S0': self.S0,
            'T': self.T,
            'cap': 0.25,
            'buffer': 0.10
        }
        
        # Simple rate curves
        r_curve = 0.02
        q_curve = 0.015
        
        option_pricer = self.create_simple_option_pricer()
        
        # Compute SCR
        scr_result = compute_one_year_scr(
            model='test_gbm',
            S_paths=S_paths,
            t_grid=t_grid,
            r_curve=r_curve,
            q_curve=q_curve,
            rila_params=rila_params,
            hedging_result=None,
            option_pricer=option_pricer
        )
        
        # Check structure
        assert 'model' in scr_result
        assert 'SCR_metrics' in scr_result
        assert 'OF_analysis' in scr_result
        assert 'raw_data' in scr_result
        
        # Check SCR metrics
        scr_metrics = scr_result['SCR_metrics']
        assert 'VaR_99.5' in scr_metrics
        assert 'CTE_99.5' in scr_metrics
        assert 'mean' in scr_metrics
        assert 'std' in scr_metrics
        
        # Check values are finite
        assert np.isfinite(scr_metrics['VaR_99.5'])
        assert np.isfinite(scr_metrics['CTE_99.5'])
        assert scr_metrics['CTE_99.5'] >= scr_metrics['VaR_99.5']
    
    def test_scr_sensitivity_to_volatility(self):
        """Test that SCR increases with higher volatility."""
        # Low volatility case
        S_paths_low, t_grid = self.generate_gbm_paths()
        
        # High volatility case (regenerate with higher vol)
        np.random.seed(789)  # Same seed for fair comparison
        dt = 1.0 / self.n_steps
        sigma_high = 0.4  # Higher volatility
        mu = 0.05
        
        dW = np.random.normal(0, np.sqrt(dt), (self.n_paths, self.n_steps))
        S_paths_high = np.zeros((self.n_paths, self.n_steps + 1))
        S_paths_high[:, 0] = self.S0
        
        for i in range(self.n_steps):
            S_paths_high[:, i+1] = S_paths_high[:, i] * np.exp(
                (mu - 0.5*sigma_high**2)*dt + sigma_high*dW[:, i]
            )
        
        # RILA parameters
        rila_params = {
            'S0': self.S0,
            'T': self.T,
            'cap': 0.25,
            'buffer': 0.10
        }
        
        r_curve = 0.02
        q_curve = 0.015
        
        # Compute SCR for both cases
        scr_low = compute_one_year_scr(
            model='low_vol', S_paths=S_paths_low, t_grid=t_grid,
            r_curve=r_curve, q_curve=q_curve, rila_params=rila_params,
            hedging_result=None, option_pricer=self.create_simple_option_pricer(0.2)
        )
        
        scr_high = compute_one_year_scr(
            model='high_vol', S_paths=S_paths_high, t_grid=t_grid,
            r_curve=r_curve, q_curve=q_curve, rila_params=rila_params,
            hedging_result=None, option_pricer=self.create_simple_option_pricer(0.4)
        )
        
        # Higher volatility should generally lead to higher SCR
        # (though this relationship can be complex for RILA products)
        assert scr_high['OF_analysis']['std_delta_OF'] >= scr_low['OF_analysis']['std_delta_OF']
    
    def test_scr_quantile_ordering(self):
        """Test that risk metrics follow expected ordering."""
        S_paths, t_grid = self.generate_gbm_paths()
        
        rila_params = {
            'S0': self.S0,
            'T': self.T,
            'cap': 0.25,
            'buffer': 0.10
        }
        
        scr_result = compute_one_year_scr(
            model='test_ordering', S_paths=S_paths, t_grid=t_grid,
            r_curve=0.02, q_curve=0.015, rila_params=rila_params,
            hedging_result=None, option_pricer=self.create_simple_option_pricer()
        )
        
        scr_metrics = scr_result['SCR_metrics']
        
        # Check quantile ordering
        if 'VaR_95.0' in scr_metrics:
            assert scr_metrics['VaR_99.5'] >= scr_metrics['VaR_99.0']
            assert scr_metrics['VaR_99.0'] >= scr_metrics['VaR_95.0']
        
        # CTE should be higher than corresponding VaR
        assert scr_metrics['CTE_99.5'] >= scr_metrics['VaR_99.5']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])