import numpy as np
import pandas as pd
from scipy.optimize import minimize
from heston_pricing_carr_madan import carr_madan_call_price
from bs_utils import bs_implied_vol

# ✅ LOAD MARKET SNAPSHOT
snapshot_date = '2018-06-01'
market_data = pd.read_csv(f'Data/SPX_Snapshot_{snapshot_date}.csv')

# ✅ FIXED GLOBAL PARAMETERS
S0 = 4500
r = 0.02
q = 0.01

# ✅ LOSS FUNCTION
def calibration_loss(params, data):
    v0, kappa, theta, sigma_v, rho = params

    # Penalize unphysical parameter regions
    if (v0 < 0 or v0 > 2 or
        kappa <= 0 or kappa > 10 or
        theta < 0 or theta > 2 or
        sigma_v <= 0 or sigma_v > 5 or
        not (-0.999 < rho < 0.999)):
        return 1e6

    errors = []

    for _, row in data.iterrows():
        K = row['strike']
        T = row['maturity_days'] / 365.0
        market_iv = row['impl_volatility']

        try:
            model_price = carr_madan_call_price(
                S0, K, T, r, q,
                v0, kappa, theta, sigma_v, rho,
                alpha=2.5,    # More stable
                u_max=100,    # Tighter integration grid
                N=1000
            )

            model_iv = bs_implied_vol(model_price, S0, K, T, r, q)

            # Handle failures or nonsense results
            if np.isnan(model_iv) or model_iv < 0.0001 or model_iv > 5:
                error = 1e2
            else:
                error = (model_iv - market_iv) ** 2

        except Exception as e:
            error = 1e2  # Penalize crashes

        errors.append(error)

    mean_error = np.mean(errors)
    print(f"✅ Loss: {mean_error:.6f} for params: {params}")
    return mean_error

# ✅ INITIAL GUESS
initial_params = [0.04, 2.0, 0.04, 0.3, -0.7]

# ✅ BOUNDS
bounds = [
    (0.01, 0.2),     # v0
    (0.1, 5.0),      # kappa
    (0.01, 0.2),     # theta
    (0.01, 1.0),     # sigma_v
    (-0.99, 0.0),    # rho — negative is realistic
]

# ✅ RUN OPTIMIZER
result = minimize(
    calibration_loss,
    initial_params,
    args=(market_data,),
    bounds=bounds,
    method='L-BFGS-B',
    options={'disp': True, 'maxiter': 200}
)

# ✅ PRINT RESULTS
print("\n✅ Calibration complete!")
print("Calibrated parameters:")
print(f"v0:      {result.x[0]:.4f}")
print(f"kappa:   {result.x[1]:.4f}")
print(f"theta:   {result.x[2]:.4f}")
print(f"sigma_v: {result.x[3]:.4f}")
print(f"rho:     {result.x[4]:.4f}")

# ✅ SAVE TO CSV
out_df = pd.DataFrame([{
    'v0': result.x[0],
    'kappa': result.x[1],
    'theta': result.x[2],
    'sigma_v': result.x[3],
    'rho': result.x[4]
}])

out_df.to_csv(f'Output/heston_calibrated_params_{snapshot_date}.csv', index=False)
print(f"✅ Saved to Output/heston_calibrated_params_{snapshot_date}.csv")
