import numpy as np
from scipy.special import roots_laguerre

def heston_characteristic_function(u, T, S0, r, q, v0, kappa, theta, sigma_v, rho):
    iu = 1j * u
    a = sigma_v**2 * (u**2 + iu)
    b = kappa - rho * sigma_v * iu
    d = np.sqrt(b**2 - 4 * a * 0.5)
    g = (b - d) / (b + d)
    exp_dT = np.exp(-d * T)
    C = r * iu * T + (kappa * theta / sigma_v**2) * ((b - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g)))
    D = (b - d) / sigma_v**2 * ((1 - exp_dT) / (1 - g * exp_dT))
    phi = np.exp(C + D * v0 + iu * np.log(S0))
    if not np.all(np.isfinite(phi)):
        print(f"[HESTON CF WARNING] Non-finite phi for params: S0={S0}, T={T}, v0={v0}, kappa={kappa}, theta={theta}, sigma_v={sigma_v}, rho={rho}")
        return np.zeros_like(u)
    return phi

def heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho, N=64):
    # Gauss-Laguerre quadrature points and weights
    x, w = roots_laguerre(N)
    def integrand(phi, Pnum):
        u = phi - 1j * 0.5
        b = kappa - (rho * sigma_v if Pnum == 1 else 0)
        d = np.sqrt((rho * sigma_v * 1j * phi - b)**2 + (sigma_v**2) * (1j * phi + phi**2))
        g = (b - d) / (b + d)
        C = r * 1j * phi * T + (kappa * theta / sigma_v**2) * ((b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = ((b - d) / sigma_v**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        f = np.exp(C + D * v0 + 1j * phi * np.log(S0 * np.exp((r - q) * T) / K))
        return np.real(np.exp(-1j * phi * np.log(K)) * f / (1j * phi))
    P1 = 0.5 + (1/np.pi) * np.sum(w * integrand(x, 1))
    P2 = 0.5 + (1/np.pi) * np.sum(w * integrand(x, 2))
    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return np.maximum(call, 0.0)

if __name__ == "__main__":
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
    print(f"Test characteristic function output: {val}")
