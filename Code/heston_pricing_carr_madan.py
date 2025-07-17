import numpy as np
from scipy.integrate import simpson
from heston_pricing_utils import heston_characteristic_function

def carr_madan_call_price(
    S0, K, T, r, q,
    v0, kappa, theta, sigma_v, rho,
    alpha=2.0, N=2000, u_max=200
):
    """
    Computes European call price using Carr-Madan Fourier method.
    """
    # Log-strike
    k = np.log(K)

    # Integration grid
    u = np.linspace(1e-5, u_max, N)

    # Characteristic function
    phi = heston_characteristic_function(u - 1j * alpha, T, S0, r, q, v0, kappa, theta, sigma_v, rho)

    # Integrand
    numerator = np.exp(-1j * u * k) * phi
    denominator = alpha**2 + alpha - u**2 + 1j * (2*alpha + 1)*u
    integrand = np.real(numerator / denominator)

    # Integration using Simpson's rule
    integral = simpson(integrand, u)

    # Price
    price = np.exp(-r*T) * integral / np.pi
    return max(price, 0.0)

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
    print(f"✅ Test Heston price (Carr-Madan): {price:.2f}")
