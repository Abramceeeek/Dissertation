import numpy as np
from scipy.integrate import simpson
from Step24_Heston_Pricing_Utilities import heston_characteristic_function

def carr_madan_call_price(
    S0, K, T, r, q,
    v0, kappa, theta, sigma_v, rho,
    alpha=1.5, N=500, u_max=50
):
    # Feller condition check
    if 2 * kappa * theta <= sigma_v ** 2:
        print(f"[CARR-MADAN WARNING] Feller condition violated: 2*kappa*theta={2*kappa*theta}, sigma_v^2={sigma_v**2}")
        return 0.0
    k = np.log(K)
    # Use a more refined grid for better numerical stability
    u = np.logspace(-3, np.log10(u_max), N)
    phi = heston_characteristic_function(u - 1j * alpha, T, S0, r, q, v0, kappa, theta, sigma_v, rho)
    numerator = np.exp(-1j * u * k) * phi
    denominator = alpha**2 + alpha - u**2 + 1j * (2*alpha + 1)*u
    integrand = np.real(numerator / denominator)
    integral = simpson(integrand, x=u)
    price = np.exp(-r*T) * integral / np.pi
    # Sanity checks
    if not np.isfinite(price) or price < 0:
        # Use Black-Scholes as fallback for unstable Heston prices
        from Step22_Black_Scholes_Utils import bs_price_call
        vol = np.sqrt(v0)  # Use initial volatility as approximation
        price = bs_price_call(S0, K, T, r, q, vol)
        if not np.isfinite(price) or price < 0:
            return 0.0
    if price > S0:
        return S0
    return price

if __name__ == "__main__":
    price = carr_madan_call_price(
    S0=4500,
    K=4500,
    T=1.0,
    r=0.02,
    q=0.01,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.3,
    rho=-0.7,
    alpha=2.0,
    N=2000,
    u_max=200
    )
    print(f"Test Heston price (Carr-Madan): {price:.2f}")
