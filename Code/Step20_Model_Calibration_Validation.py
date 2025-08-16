import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from Step23_Heston_Carr_Madan_Pricing import carr_madan_call_price
from Step22_Black_Scholes_Utils import bs_implied_vol, bs_price_call
from Step24_Heston_Pricing_Utilities import heston_characteristic_function, heston_call_price

def validate_heston_calibration(snapshot_date='2018-06-01', params_file=None, use_stable_heston=True):
    print(f"Validating Heston calibration for {snapshot_date}... (Stable pricer: {use_stable_heston})")
    
    try:
        market_data = pd.read_csv(f'Data/SPX_Snapshot_{snapshot_date}.csv')
        print(f"  Loaded {len(market_data)} market options")
    except FileNotFoundError:
        print(f"Could not find market snapshot: Data/SPX_Snapshot_{snapshot_date}.csv")
        return None
    
    if params_file is None:
        params_file = f'Output/heston_calibrated_params_{snapshot_date}_DE.csv'
    
    try:
        params_df = pd.read_csv(params_file)
        params = params_df.iloc[0]  
        v0, kappa, theta, sigma_v, rho = params['v0'], params['kappa'], params['theta'], params['sigma_v'], params['rho']
        print(f"  Loaded parameters: v0={v0:.4f}, kappa={kappa:.4f}, theta={theta:.4f}, sigma_v={sigma_v:.4f}, rho={rho:.4f}")
    except FileNotFoundError:
        print(f"Could not find calibrated parameters: {params_file}")
        return None
    
    S0 = 4500
    r = 0.02
    q = 0.01
    
    model_ivs = []
    market_ivs = []
    strikes = []
    maturities = []
    errors = []
    
    print("  Computing model implied volatilities...")
    
    debug_count = 0
    for idx, row in market_data.iterrows():
        if idx % 100 == 0:
            print(f"    Progress: {idx}/{len(market_data)}")
        # Fix strike scaling
        if 'strike' in row and abs(row['strike'] - 4500) < 10000:
            K = row['strike']
        elif 'strike_price' in row:
            K = row['strike_price'] / 1000  # Convert to human scale
            if idx == 0:
                print(f"[WARNING] Using strike_price/1000 for K: {K}")
        else:
            print(f"[ERROR] No valid strike column found in row {idx}")
            continue
        T = row['maturity_days'] / 365.0
        market_iv = row['impl_volatility']
        if T < 0.01 or T > 3.0:
            continue
        moneyness = K / S0
        if moneyness < 0.7 or moneyness > 1.4:
            continue
        try:
            if use_stable_heston:
                model_price = heston_call_price(
                    S0, K, T, r, q, v0, kappa, theta, sigma_v, rho
                )
            else:
                model_price = carr_madan_call_price(
                    S0, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                    alpha=2.5, u_max=80, N=800
                )
            intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0)
            if model_price < intrinsic:
                if debug_count < 10:
                    print(f"[WARNING] Model price below intrinsic: idx={idx}, K={K}, T={T:.4f}, model_price={model_price:.6f}, intrinsic={intrinsic:.6f}")
                    debug_count += 1
                model_iv = np.nan
            else:
                model_iv = bs_implied_vol(model_price, S0, K, T, r, q)
            if not np.isnan(model_iv) and 0.05 < model_iv < 1.0:
                model_ivs.append(model_iv)
                market_ivs.append(market_iv)
                strikes.append(K)
                maturities.append(T)
                errors.append((model_iv - market_iv)**2)
        except Exception as e:
            if debug_count < 10:
                print(f"[DEBUG-EXCEPTION] idx={idx}, K={K}, T={T:.4f}, Exception: {e}")
                debug_count += 1
            continue
    
    if len(model_ivs) == 0:
        print("No valid model prices computed")
        return None
    
    model_ivs = np.array(model_ivs)
    market_ivs = np.array(market_ivs)
    strikes = np.array(strikes)
    maturities = np.array(maturities)
    errors = np.array(errors)
    
    print(f"  Successfully computed {len(model_ivs)} model implied volatilities")
    
    rmse = np.sqrt(np.mean(errors))
    mae = np.mean(np.abs(model_ivs - market_ivs))
    max_error = np.max(np.abs(model_ivs - market_ivs))
    r_squared = 1 - np.sum(errors) / np.sum((market_ivs - np.mean(market_ivs))**2)
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Max Error: {max_error:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    
    os.makedirs('Output/plots', exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(market_ivs, model_ivs, alpha=0.6, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit')
    plt.xlabel('Market Implied Volatility')
    plt.ylabel('Model Implied Volatility')
    plt.title(f'Model vs Market IV\n{snapshot_date}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}\nRMSE = {rmse:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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
    
    unique_maturities = np.unique(np.round(maturities * 365))
    if len(unique_maturities) >= 3:
        plot_iv_surface_comparison(strikes, maturities, market_ivs, model_ivs, S0, snapshot_date)
    
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
    
    ax1 = fig.add_subplot(121, projection='3d')
    moneyness = strikes / S0
    scatter1 = ax1.scatter(moneyness, maturities, market_ivs, c=market_ivs, cmap='viridis', s=20)
    ax1.set_xlabel('Moneyness (K/S)')
    ax1.set_ylabel('Time to Maturity (years)')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title(f'Market IV Surface\n{snapshot_date}')
    
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
    print(f"Analyzing parameter sensitivity for {snapshot_date}...")
    
    base_results = validate_heston_calibration(snapshot_date, base_params_file)
    if base_results is None:
        return None
    
    base_params = base_results['parameters']
    base_rmse = base_results['rmse']
    
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
            perturbed_params = base_params.copy()
            perturbed_params[param_name] += delta
            
            if (perturbed_params['v0'] <= 0 or perturbed_params['kappa'] <= 0 or 
                perturbed_params['theta'] <= 0 or perturbed_params['sigma_v'] <= 0 or
                abs(perturbed_params['rho']) >= 1):
                continue
            param_values.append(perturbed_params[param_name])
        
        sensitivity_results[param_name] = {
            'values': param_values,
            'base_value': base_params[param_name],
            'base_rmse': base_rmse
        }
    
    return sensitivity_results

def validate_black_scholes(snapshot_date='2018-06-01'):
    print(f"Validating Black-Scholes model for {snapshot_date}...")
    try:
        market_data = pd.read_csv(f'Data/SPX_Snapshot_{snapshot_date}.csv')
        print(f"  Loaded {len(market_data)} market options")
    except FileNotFoundError:
        print(f"Could not find market snapshot: Data/SPX_Snapshot_{snapshot_date}.csv")
        return None
    S0 = 4500
    r = 0.02
    q = 0.01
    sigma = 0.2  # Use a fixed volatility for demonstration
    model_ivs = []
    market_ivs = []
    strikes = []
    maturities = []
    errors = []
    for idx, row in market_data.iterrows():
        if 'strike' in row and abs(row['strike'] - 4500) < 10000:
            K = row['strike']
        elif 'strike_price' in row:
            K = row['strike_price'] / 1000
        else:
            continue
        T = row['maturity_days'] / 365.0
        market_iv = row['impl_volatility']
        if T < 0.01 or T > 3.0:
            continue
        moneyness = K / S0
        if moneyness < 0.7 or moneyness > 1.4:
            continue
        try:
            model_price = bs_price_call(S0, K, T, r, q, sigma)
            model_iv = bs_implied_vol(model_price, S0, K, T, r, q)
            if not np.isnan(model_iv) and 0.05 < model_iv < 1.0:
                model_ivs.append(model_iv)
                market_ivs.append(market_iv)
                strikes.append(K)
                maturities.append(T)
                errors.append((model_iv - market_iv)**2)
        except Exception as e:
            continue
    if len(model_ivs) == 0:
        print("No valid Black-Scholes model prices computed")
        return None
    model_ivs = np.array(model_ivs)
    market_ivs = np.array(market_ivs)
    strikes = np.array(strikes)
    maturities = np.array(maturities)
    errors = np.array(errors)
    print(f"  Successfully computed {len(model_ivs)} Black-Scholes model implied volatilities")
    rmse = np.sqrt(np.mean(errors))
    mae = np.mean(np.abs(model_ivs - market_ivs))
    max_error = np.max(np.abs(model_ivs - market_ivs))
    r_squared = 1 - np.sum(errors) / np.sum((market_ivs - np.mean(market_ivs))**2)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Max Error: {max_error:.4f}")
    print(f"  R-squared: {r_squared:.4f}")

    # Plotting
    os.makedirs('Output/plots', exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(market_ivs, model_ivs, alpha=0.6, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit')
    plt.xlabel('Market Implied Volatility')
    plt.ylabel('Model Implied Volatility')
    plt.title(f'BS Model vs Market IV\n{snapshot_date}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}\nRMSE = {rmse:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.subplot(1, 3, 2)
    moneyness = strikes / S0
    iv_errors = model_ivs - market_ivs
    plt.scatter(moneyness, iv_errors, alpha=0.6, s=20, c=maturities, cmap='viridis')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Moneyness (K/S)')
    plt.ylabel('IV Error (Model - Market)')
    plt.title('BS Calibration Errors vs Moneyness')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Time to Maturity (years)')
    plt.subplot(1, 3, 3)
    plt.scatter(maturities, iv_errors, alpha=0.6, s=20, c=moneyness, cmap='plasma')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('IV Error (Model - Market)')
    plt.title('BS Calibration Errors vs Maturity')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Moneyness (K/S)')
    plt.tight_layout()
    plt.savefig(f'Output/plots/bs_calibration_validation_{snapshot_date}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'snapshot_date': snapshot_date,
        'n_options': len(model_ivs),
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'r_squared': r_squared,
        'market_ivs': market_ivs,
        'model_ivs': model_ivs,
        'strikes': strikes,
        'maturities': maturities
    }

if __name__ == "__main__":
    print("Starting Heston Calibration Validation")
    print("="*50)
    
    snapshot_date = '2018-06-01'
    for method in ['DE', '']:
        params_file = f'Output/heston_calibrated_params_{snapshot_date}{"_" + method if method else ""}.csv'
        print(f"\nValidating {method if method else 'Local'} optimization results...")
        results = validate_heston_calibration(snapshot_date, params_file, use_stable_heston=True)
        if results is not None:
            print(f"{method if method else 'Local'} calibration validation completed")
            print(f"   Model fit quality: RMSE = {results['rmse']:.4f}, R² = {results['r_squared']:.4f}")
        else:
            print(f"{method if method else 'Local'} calibration validation failed")
    print(f"\nCalibration validation completed!")
    print(f"Validation plots saved to Output/plots/")

    # Test Carr-Madan pricer for a typical ATM option with DE params
    import pandas as pd
    params_df = pd.read_csv('Output/heston_calibrated_params_2018-06-01_DE.csv')
    params = params_df.iloc[0]
    v0, kappa, theta, sigma_v, rho = params['v0'], params['kappa'], params['theta'], params['sigma_v'], params['rho']
    from heston_pricing_carr_madan import carr_madan_call_price
    price = carr_madan_call_price(
        S0=4500, K=4500, T=1.0, r=0.02, q=0.01,
        v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho
    )
    print(f"[TEST] Carr-Madan price for ATM option (DE params): {price}")

    # Print characteristic function output for ATM test case
    from heston_pricing_utils import heston_characteristic_function
    phi = heston_characteristic_function(
        u=1.0,
        T=1.0,
        S0=4500,
        r=0.02,
        q=0.01,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho
    )
    print(f"[TEST] Heston characteristic function output (u=1.0, ATM, DE params): {phi}")

    print(f"\n[BLACK-SCHOLES VALIDATION]")
    bs_results = validate_black_scholes('2018-06-01')
    if bs_results is not None:
        print(f"Black-Scholes validation completed: RMSE = {bs_results['rmse']:.4f}, R² = {bs_results['r_squared']:.4f}")
    else:
        print("Black-Scholes validation failed")