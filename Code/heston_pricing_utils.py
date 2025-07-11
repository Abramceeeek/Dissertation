import numpy as np

def heston_characteristic_function(u, T, S0, r, q, v0, kappa, theta, sigma_v, rho):
    """
    Computes the Heston characteristic function phi(u).
    """
    iu = 1j * u
    a = sigma_v**2 * (u**2 + iu)
    b = kappa - rho * sigma_v * iu
    d = np.sqrt(b**2 - 4 * a * 0.5)
    g = (b - d) / (b + d)

    exp_dT = np.exp(-d * T)
    C = r * iu * T + (kappa * theta / sigma_v**2) * ((b - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g)))
    D = (b - d) / sigma_v**2 * ((1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0 + iu * np.log(S0))

if __name__ == "__main__":
    # Quick test
    val = heston_characteristic_function(
        u=1.0,
        T=1.0,
        S0=4500,
        r=0.02,
        q=0.01,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7
    )
    print(f"✅ Test characteristic function output: {val}")
