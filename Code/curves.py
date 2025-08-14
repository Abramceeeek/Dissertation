"""
Interest Rate and Dividend Yield Curve Management

This module provides functionality to load, interpolate and work with 
risk-free rate curves and dividend yield curves for RILA pricing and SCR calculations.

Author: Abdurakhmonbek Fayzullaev
Purpose: MSc Dissertation - Solvency II SCR for Equity-Linked Variable Annuities
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Union, Optional, Tuple
from scipy.interpolate import interp1d, CubicSpline
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YieldCurve:
    """
    Yield curve class for interest rates and dividend yields with interpolation capabilities.
    """
    
    def __init__(self, maturities: np.ndarray, rates: np.ndarray, 
                 interpolation: str = 'linear', extrapolation: str = 'flat'):
        """
        Initialize yield curve.
        
        Args:
            maturities (np.ndarray): Time to maturity in years
            rates (np.ndarray): Corresponding rates (as decimals, e.g., 0.05 for 5%)
            interpolation (str): Interpolation method ('linear', 'cubic', 'log-linear')
            extrapolation (str): Extrapolation method ('flat', 'linear')
        """
        self.maturities = np.array(maturities)
        self.rates = np.array(rates)
        self.interpolation = interpolation
        self.extrapolation = extrapolation
        
        # Sort by maturity
        sort_idx = np.argsort(self.maturities)
        self.maturities = self.maturities[sort_idx]
        self.rates = self.rates[sort_idx]
        
        # Create interpolator
        self._create_interpolator()
        
        logger.debug(f"Created yield curve with {len(self.maturities)} points, "
                    f"maturity range: {self.maturities[0]:.3f} - {self.maturities[-1]:.3f} years")
    
    def _create_interpolator(self):
        """Create the interpolation function."""
        if len(self.maturities) < 2:
            # Constant curve
            self._interpolator = lambda t: np.full_like(t, self.rates[0])
            return
        
        if self.interpolation == 'linear':
            bounds_error = (self.extrapolation != 'linear')
            fill_value = 'extrapolate' if self.extrapolation == 'linear' else (self.rates[0], self.rates[-1])
            self._interpolator = interp1d(
                self.maturities, self.rates, kind='linear',
                bounds_error=bounds_error, fill_value=fill_value
            )
        elif self.interpolation == 'cubic':
            if len(self.maturities) >= 4:
                self._interpolator = CubicSpline(self.maturities, self.rates, bc_type='natural')
            else:
                # Fall back to linear for insufficient points
                logger.warning("Insufficient points for cubic interpolation, using linear")
                self._interpolator = interp1d(
                    self.maturities, self.rates, kind='linear',
                    bounds_error=False, fill_value=(self.rates[0], self.rates[-1])
                )
        elif self.interpolation == 'log-linear':
            # Log-linear interpolation in discount factors
            discount_factors = np.exp(-self.rates * self.maturities)
            df_interpolator = interp1d(
                self.maturities, discount_factors, kind='linear',
                bounds_error=False, fill_value=(discount_factors[0], discount_factors[-1])
            )
            def log_linear_rate(t):
                t = np.atleast_1d(t)
                df = df_interpolator(t)
                # Avoid division by zero
                rates = np.where(t > 1e-6, -np.log(df) / t, self.rates[0])
                return rates if len(rates) > 1 else rates[0]
            self._interpolator = log_linear_rate
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")
    
    def __call__(self, maturity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get interpolated rate(s) for given maturity/maturities.
        
        Args:
            maturity: Time to maturity in years (scalar or array)
            
        Returns:
            Interpolated rate(s)
        """
        maturity = np.atleast_1d(maturity)
        rates = self._interpolator(maturity)
        
        # Handle extrapolation
        if self.extrapolation == 'flat':
            rates = np.where(maturity < self.maturities[0], self.rates[0], rates)
            rates = np.where(maturity > self.maturities[-1], self.rates[-1], rates)
        
        return rates if len(rates) > 1 else rates[0]
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Calculate forward rate from t1 to t2.
        
        Args:
            t1 (float): Start time
            t2 (float): End time
            
        Returns:
            float: Forward rate
        """
        if t2 <= t1:
            return self(t1)
        
        r1, r2 = self(t1), self(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)
    
    def discount_factor(self, maturity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate discount factor(s) for given maturity/maturities.
        
        Args:
            maturity: Time to maturity in years
            
        Returns:
            Discount factor(s)
        """
        rates = self(maturity)
        maturity = np.atleast_1d(maturity)
        df = np.exp(-rates * maturity)
        return df if len(df) > 1 else df[0]
    
    def summary(self) -> Dict:
        """Get curve summary statistics."""
        return {
            'min_maturity': float(self.maturities[0]),
            'max_maturity': float(self.maturities[-1]),
            'min_rate': float(np.min(self.rates)),
            'max_rate': float(np.max(self.rates)),
            'avg_rate': float(np.mean(self.rates)),
            'num_points': len(self.maturities),
            'interpolation': self.interpolation
        }

def load_risk_free_curve(file_path: str, date: str = None, 
                        interpolation: str = 'linear') -> YieldCurve:
    """
    Load risk-free rate curve from CSV file.
    
    Args:
        file_path (str): Path to CSV file with rate data
        date (str, optional): Specific date to load (YYYY-MM-DD format)
        interpolation (str): Interpolation method
        
    Returns:
        YieldCurve: Risk-free rate curve
    """
    logger.info(f"Loading risk-free curve from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Loaded file with columns: {list(df.columns)}")
        
        # Handle different possible column names
        date_col = None
        for col in ['date', 'Date', 'DATE']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col and date:
            # Filter by specific date
            df['date_parsed'] = pd.to_datetime(df[date_col])
            target_date = pd.to_datetime(date)
            df_filtered = df[df['date_parsed'] == target_date]
            
            if df_filtered.empty:
                # Find closest date
                df['date_diff'] = np.abs((df['date_parsed'] - target_date).dt.days)
                closest_idx = df['date_diff'].idxmin()
                df_filtered = df.iloc[[closest_idx]]
                actual_date = df.iloc[closest_idx][date_col]
                logger.warning(f"Date {date} not found, using closest date: {actual_date}")
            
            df = df_filtered
        
        # Extract maturity and rate columns
        # Look for standard Treasury maturity columns
        maturity_columns = []
        for col in df.columns:
            if any(term in col.lower() for term in ['mo', 'yr', 'month', 'year']) and \
               any(char.isdigit() for char in col):
                maturity_columns.append(col)
        
        if not maturity_columns:
            # Try to find numeric columns (excluding date)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            maturity_columns = [col for col in numeric_cols if col not in [date_col]]
        
        logger.info(f"Found maturity columns: {maturity_columns}")
        
        # Parse maturities and rates
        maturities = []
        rates = []
        
        for col in maturity_columns:
            # Parse maturity from column name
            col_lower = col.lower()
            if 'mo' in col_lower or 'month' in col_lower:
                # Extract number and convert months to years
                maturity_num = float(''.join(c for c in col if c.isdigit() or c == '.'))
                maturity_years = maturity_num / 12
            elif 'yr' in col_lower or 'year' in col_lower:
                # Extract number for years
                maturity_num = float(''.join(c for c in col if c.isdigit() or c == '.'))
                maturity_years = maturity_num
            else:
                # Try to infer from column name or use default mapping
                try:
                    maturity_years = float(''.join(c for c in col if c.isdigit() or c == '.'))
                    if maturity_years > 50:  # Assume it's in months if > 50
                        maturity_years /= 12
                except:
                    logger.warning(f"Could not parse maturity from column: {col}")
                    continue
            
            # Get rate value (use last available value)
            rate_value = df[col].iloc[-1] if not df[col].empty else np.nan
            
            if not np.isnan(rate_value) and maturity_years > 0:
                maturities.append(maturity_years)
                rates.append(rate_value / 100 if rate_value > 1 else rate_value)  # Convert % to decimal
        
        if not maturities:
            raise ValueError("No valid maturity/rate data found in file")
        
        curve = YieldCurve(np.array(maturities), np.array(rates), interpolation=interpolation)
        
        logger.info(f"Created risk-free curve with {len(maturities)} points")
        logger.info(f"Maturity range: {min(maturities):.2f} - {max(maturities):.2f} years")
        logger.info(f"Rate range: {min(rates):.2%} - {max(rates):.2%}")
        
        return curve
        
    except Exception as e:
        logger.error(f"Error loading risk-free curve: {e}")
        # Return flat curve as fallback
        logger.warning("Returning flat 2% risk-free curve as fallback")
        return YieldCurve(np.array([0.25, 1, 2, 5, 10]), np.array([0.02, 0.02, 0.02, 0.02, 0.02]))

def load_dividend_yield_curve(file_path: str, date: str = None) -> YieldCurve:
    """
    Load dividend yield curve from CSV file.
    
    Args:
        file_path (str): Path to CSV file with dividend yield data
        date (str, optional): Specific date to load
        
    Returns:
        YieldCurve: Dividend yield curve
    """
    logger.info(f"Loading dividend yield curve from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Handle date filtering if specified
        if date and 'date' in df.columns:
            df['date_parsed'] = pd.to_datetime(df['date'])
            target_date = pd.to_datetime(date)
            df_filtered = df[df['date_parsed'] == target_date]
            
            if df_filtered.empty:
                # Use last available date
                df_filtered = df.iloc[[-1]]
                logger.warning(f"Date {date} not found, using last available date")
            
            df = df_filtered
        
        # Look for dividend yield column
        div_yield_col = None
        for col in df.columns:
            if any(term in col.lower() for term in ['div', 'yield', 'dividend']):
                div_yield_col = col
                break
        
        if div_yield_col is None:
            # Use last numeric column as fallback
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            div_yield_col = numeric_cols[-1] if len(numeric_cols) > 0 else None
        
        if div_yield_col is None:
            raise ValueError("No dividend yield column found")
        
        # Get dividend yield value
        div_yield = df[div_yield_col].iloc[-1]
        div_yield = div_yield / 100 if div_yield > 1 else div_yield  # Convert % to decimal
        
        # Create flat dividend yield curve
        maturities = np.array([0.25, 0.5, 1, 2, 5, 10])
        div_yields = np.full_like(maturities, div_yield)
        
        curve = YieldCurve(maturities, div_yields, interpolation='linear')
        
        logger.info(f"Created dividend yield curve: {div_yield:.2%} (flat)")
        return curve
        
    except Exception as e:
        logger.error(f"Error loading dividend yield curve: {e}")
        # Return flat 1.5% dividend yield as fallback
        logger.warning("Returning flat 1.5% dividend yield curve as fallback")
        maturities = np.array([0.25, 0.5, 1, 2, 5, 10])
        div_yields = np.full_like(maturities, 0.015)
        return YieldCurve(maturities, div_yields)

def create_curve_from_data(maturities: list, rates: list, 
                          interpolation: str = 'linear') -> YieldCurve:
    """
    Create yield curve from maturity and rate lists.
    
    Args:
        maturities (list): Maturities in years
        rates (list): Corresponding rates (as decimals)
        interpolation (str): Interpolation method
        
    Returns:
        YieldCurve: Constructed yield curve
    """
    return YieldCurve(np.array(maturities), np.array(rates), interpolation=interpolation)

def get_market_curves(date: str = '2021-06-01', 
                     risk_free_file: str = 'Data/Risk-Free Yield Curve/Interest_Rate_Curves_2018_2023_CLEANED.csv',
                     dividend_file: str = 'Data/Dividend Yield Data/SPX_Implied_Yield_Rates_2018_2023.csv') -> Tuple[YieldCurve, YieldCurve]:
    """
    Load both risk-free and dividend yield curves for a specific date.
    
    Args:
        date (str): Date in YYYY-MM-DD format
        risk_free_file (str): Path to risk-free rate file
        dividend_file (str): Path to dividend yield file
        
    Returns:
        Tuple[YieldCurve, YieldCurve]: (risk_free_curve, dividend_curve)
    """
    logger.info(f"Loading market curves for date: {date}")
    
    # Check if files exist
    if not os.path.exists(risk_free_file):
        logger.warning(f"Risk-free file not found: {risk_free_file}")
    if not os.path.exists(dividend_file):
        logger.warning(f"Dividend file not found: {dividend_file}")
    
    r_curve = load_risk_free_curve(risk_free_file, date) if os.path.exists(risk_free_file) else \
              create_curve_from_data([0.25, 1, 2, 5, 10], [0.02, 0.02, 0.02, 0.02, 0.02])
    
    q_curve = load_dividend_yield_curve(dividend_file, date) if os.path.exists(dividend_file) else \
              create_curve_from_data([0.25, 1, 2, 5, 10], [0.015, 0.015, 0.015, 0.015, 0.015])
    
    return r_curve, q_curve

# Example usage and testing
if __name__ == "__main__":
    # Test yield curve functionality
    print("Testing YieldCurve functionality...")
    
    # Create sample curve
    maturities = np.array([0.25, 0.5, 1, 2, 5, 10])
    rates = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
    
    curve = YieldCurve(maturities, rates, interpolation='linear')
    
    # Test interpolation
    test_maturities = np.array([0.1, 0.75, 1.5, 3, 7, 15])
    interpolated_rates = curve(test_maturities)
    
    print("Interpolation test:")
    for t, r in zip(test_maturities, interpolated_rates):
        print(f"  T={t:.2f}y: {r:.3%}")
    
    # Test discount factors
    print(f"\nDiscount factors:")
    for t in [0.5, 1, 2, 5]:
        df = curve.discount_factor(t)
        print(f"  T={t}y: DF={df:.6f}")
    
    # Test curve summary
    print(f"\nCurve summary: {curve.summary()}")
    
    # Test curve loading (if files exist)
    try:
        rf_curve, div_curve = get_market_curves('2021-06-01')
        print(f"\nRisk-free curve summary: {rf_curve.summary()}")
        print(f"Dividend curve summary: {div_curve.summary()}")
        
        # Test rate values
        print(f"\nSample rates:")
        for t in [0.5, 1, 2, 5]:
            rf_rate = rf_curve(t)
            div_rate = div_curve(t)
            print(f"  T={t}y: r={rf_rate:.3%}, q={div_rate:.3%}")
            
    except Exception as e:
        print(f"Could not load market curves: {e}")
        print("Using synthetic curves for testing")