import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from heston_pricing_carr_madan import carr_madan_call_price
from bs_utils import bs_implied_vol

def validate_heston_calibration(snapshot_date='2018-06-01', params_file=None):
    """
    Validate Heston model calibration by comparing model vs market implied volatilities.
    
    Parameters:
    - snapshot_date: date of market snapshot
    - params_file: path to calibrated parameters CSV
    
    Returns:
    - validation_results: dictionary with fit statistics and plots
    """
    print(f"🔍 Validating Heston calibration for {snapshot_date}...")
    
    # Load market data
    try:
        market_data = pd.read_csv(f'Data/SPX_Snapshot_{snapshot_date}.csv')
        print(f"  Loaded {len(market_data)} market options")
    except FileNotFoundError:
        print(f"❌ Could not find market snapshot: Data/SPX_Snapshot_{snapshot_date}.csv")
        return None
    
    # Load calibrated parameters
    if params_file is None:
        params_file = f'Output/heston_calibrated_params_{snapshot_date}_DE.csv'
    
    try:
        params_df = pd.read_csv(params_file)
        params = params_df.iloc[0]  # Assuming first row contains parameters
        v0, kappa, theta, sigma_v, rho = params['v0'], params['kappa'], params['theta'], params['sigma_v'], params['rho']
        print(f"  Loaded parameters: v0={v0:.4f}, kappa={kappa:.4f}, theta={theta:.4f}, sigma_v={sigma_v:.4f}, rho={rho:.4f}")
    except FileNotFoundError:
        print(f"❌ Could not find calibrated parameters: {params_file}")
        return None
    
    # Fixed parameters for pricing
    S0 = 4500
    r = 0.02
    q = 0.01
    
    # Calculate model implied volatilities
    model_ivs = []
    market_ivs = []
    strikes = []
    maturities = []
    errors = []
    
    print("  Computing model implied volatilities...")
    
    for idx, row in market_data.iterrows():
        if idx % 100 == 0:
            print(f"    Progress: {idx}/{len(market_data)}")
        
        K = row['strike']
        T = row['maturity_days'] / 365.0
        market_iv = row['impl_volatility']
        
        # Skip if maturity is too short or too long
        if T < 0.01 or T > 3.0:
            continue
            
        # Skip if strike is too far OTM/ITM
        moneyness = K / S0
        if moneyness < 0.7 or moneyness > 1.4:
            continue
        
        try:
            # Calculate model price
            model_price = carr_madan_call_price(
                S0, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                alpha=2.5, u_max=80, N=800
            )
            
            # Convert to implied volatility
            model_iv = bs_implied_vol(model_price, S0, K, T, r, q)
            
            # Store if valid
            if not np.isnan(model_iv) and 0.05 < model_iv < 1.0:
                model_ivs.append(model_iv)
                market_ivs.append(market_iv)
                strikes.append(K)
                maturities.append(T)
                errors.append((model_iv - market_iv)**2)
            
        except Exception as e:
            continue
    
    if len(model_ivs) == 0:
        print("❌ No valid model prices computed")
        return None
    
    # Convert to arrays
    model_ivs = np.array(model_ivs)
    market_ivs = np.array(market_ivs)
    strikes = np.array(strikes)
    maturities = np.array(maturities)
    errors = np.array(errors)
    
    print(f"  Successfully computed {len(model_ivs)} model implied volatilities")
    
    # Calculate fit statistics
    rmse = np.sqrt(np.mean(errors))
    mae = np.mean(np.abs(model_ivs - market_ivs))
    max_error = np.max(np.abs(model_ivs - market_ivs))
    r_squared = 1 - np.sum(errors) / np.sum((market_ivs - np.mean(market_ivs))**2)
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Max Error: {max_error:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    
    # Create validation plots
    os.makedirs('Output/plots', exist_ok=True)
    
    # 1. Model vs Market IV scatter plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(market_ivs, model_ivs, alpha=0.6, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit')
    plt.xlabel('Market Implied Volatility')
    plt.ylabel('Model Implied Volatility')
    plt.title(f'Model vs Market IV\n{snapshot_date}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add R-squared annotation
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}\nRMSE = {rmse:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Error vs Moneyness
    plt.subplot(1, 3, 2)
    moneyness = strikes / S0
    iv_errors = model_ivs - market_ivs
    plt.scatter(moneyness, iv_errors, alpha=0.6, s=20, c=maturities, cmap='viridis')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Moneyness (K/S)')
    plt.ylabel('IV Error (Model - Market)')
    plt.title('Calibration Errors vs Moneyness')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Time to Maturity (years)')
    
    # 3. Error vs Time to Maturity
    plt.subplot(1, 3, 3)
    plt.scatter(maturities, iv_errors, alpha=0.6, s=20, c=moneyness, cmap='plasma')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('IV Error (Model - Market)')
    plt.title('Calibration Errors vs Maturity')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Moneyness (K/S)')
    
    plt.tight_layout()
    plt.savefig(f'Output/plots/heston_calibration_validation_{snapshot_date}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. IV Surface comparison (if we have enough data points)
    unique_maturities = np.unique(np.round(maturities * 365))
    if len(unique_maturities) >= 3:
        plot_iv_surface_comparison(strikes, maturities, market_ivs, model_ivs, S0, snapshot_date)
    
    # Create results summary
    validation_results = {
        'snapshot_date': snapshot_date,
        'n_options': len(model_ivs),
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'r_squared': r_squared,
        'parameters': {
            'v0': v0, 'kappa': kappa, 'theta': theta, 
            'sigma_v': sigma_v, 'rho': rho
        },
        'market_ivs': market_ivs,
        'model_ivs': model_ivs,
        'strikes': strikes,
        'maturities': maturities
    }
    
    return validation_results

def plot_iv_surface_comparison(strikes, maturities, market_ivs, model_ivs, S0, snapshot_date):
    """
    Plot 3D comparison of market vs model IV surfaces.
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Market surface
    ax1 = fig.add_subplot(121, projection='3d')
    moneyness = strikes / S0
    scatter1 = ax1.scatter(moneyness, maturities, market_ivs, c=market_ivs, cmap='viridis', s=20)
    ax1.set_xlabel('Moneyness (K/S)')
    ax1.set_ylabel('Time to Maturity (years)')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title(f'Market IV Surface\n{snapshot_date}')
    
    # Model surface
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(moneyness, maturities, model_ivs, c=model_ivs, cmap='viridis', s=20)
    ax2.set_xlabel('Moneyness (K/S)')
    ax2.set_ylabel('Time to Maturity (years)')
    ax2.set_zlabel('Implied Volatility')
    ax2.set_title(f'Heston Model IV Surface\n{snapshot_date}')
    
    plt.tight_layout()
    plt.savefig(f'Output/plots/iv_surface_comparison_{snapshot_date}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def analyze_parameter_sensitivity(snapshot_date='2018-06-01', base_params_file=None):
    """
    Analyze sensitivity of calibration to parameter perturbations.
    """
    print(f"📊 Analyzing parameter sensitivity for {snapshot_date}...")
    
    # Load base validation results
    base_results = validate_heston_calibration(snapshot_date, base_params_file)
    if base_results is None:
        return None
    
    base_params = base_results['parameters']
    base_rmse = base_results['rmse']
    
    # Define parameter perturbations
    perturbations = {
        'v0': [-0.01, -0.005, 0.005, 0.01],
        'kappa': [-0.5, -0.2, 0.2, 0.5],
        'theta': [-0.01, -0.005, 0.005, 0.01],
        'sigma_v': [-0.05, -0.02, 0.02, 0.05],
        'rho': [-0.1, -0.05, 0.05, 0.1]
    }
    
    sensitivity_results = {}
    
    for param_name in perturbations:
        print(f"  Testing sensitivity to {param_name}...")
        param_rmses = []
        param_values = []
        
        for delta in perturbations[param_name]:
            # Create perturbed parameters
            perturbed_params = base_params.copy()
            perturbed_params[param_name] += delta
            
            # Skip if parameters become invalid
            if (perturbed_params['v0'] <= 0 or perturbed_params['kappa'] <= 0 or 
                perturbed_params['theta'] <= 0 or perturbed_params['sigma_v'] <= 0 or
                abs(perturbed_params['rho']) >= 1):
                continue
            
            # Compute RMSE with perturbed parameters
            # (This would require re-running the validation with new parameters)
            # For now, we'll store the parameter values for plotting
            param_values.append(perturbed_params[param_name])
            # param_rmses.append(computed_rmse)  # Would compute this
        
        sensitivity_results[param_name] = {
            'values': param_values,
            'base_value': base_params[param_name],
            'base_rmse': base_rmse
        }
    
    return sensitivity_results

# Main execution
if __name__ == "__main__":
    print("🔬 Starting Heston Calibration Validation")
    print("="*50)
    
    # Validate calibration for the main snapshot date
    snapshot_date = '2018-06-01'
    
    # Try both calibration methods
    for method in ['DE', '']:  # DE (Differential Evolution) and local optimization
        params_file = f'Output/heston_calibrated_params_{snapshot_date}{"_" + method if method else ""}.csv'
        
        print(f"\nValidating {method if method else 'Local'} optimization results...")
        results = validate_heston_calibration(snapshot_date, params_file)
        
        if results is not None:
            print(f"✅ {method if method else 'Local'} calibration validation completed")
            print(f"   Model fit quality: RMSE = {results['rmse']:.4f}, R² = {results['r_squared']:.4f}")
        else:
            print(f"❌ {method if method else 'Local'} calibration validation failed")
    
    print(f"\n✅ Calibration validation completed!")
    print(f"📊 Validation plots saved to Output/plots/")