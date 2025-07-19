import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from rila.models import simulate_gbm, simulate_heston, simulate_rough_vol
from rila.payoff import apply_rila_payoff
from rila.config import S0, n_paths, T, N, heston_params

st.title('RILA Risk & Hedging Interactive Dashboard')

model = st.selectbox('Model', ['GBM', 'Heston', 'Rough Vol'])
buffer = st.slider('Buffer (downside protection)', 0.0, 0.3, 0.1, 0.01)
cap = st.slider('Cap (upside limit)', 0.05, 1.0, 0.5, 0.01)
n_paths_dash = st.slider('Number of Paths', 100, 5000, 1000, 100)

if st.button('Simulate'):
    st.write(f'Simulating {model}...')
    if model == 'GBM':
        mu = 0.01
        sigma = 0.2
        S = simulate_gbm(S0, mu, sigma, T, N, n_paths_dash)
    elif model == 'Heston':
        mu = 0.01
        S = simulate_heston(S0, heston_params['v0'], mu, heston_params['kappa'], heston_params['theta'], heston_params['sigma_v'], heston_params['rho'], T, N, n_paths_dash)
    else:
        mu = 0.01
        S = simulate_rough_vol(S0, mu, 0.04, 1.5, 0.1, T, N, n_paths_dash)
    start_values = S[0]
    end_values = S[-1]
    returns = (end_values - start_values) / start_values
    returns = np.clip(returns, -0.9, 1.5)
    credited = apply_rila_payoff(returns, buffer, cap)
    final_accounts = S0 * (1 + credited)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(final_accounts, bins=50, alpha=0.7, edgecolor='black')
    ax.set_title(f'Final Account Value Distribution ({model})')
    ax.set_xlabel('Account Value')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write(f"Mean: {np.mean(final_accounts):.2f}")
    st.write(f"Std: {np.std(final_accounts):.2f}")
    st.write(f"5% VaR: {np.percentile(final_accounts, 5):.2f}")
    st.write(f"Worst Case: {np.min(final_accounts):.2f}")
    st.write(f"Best Case: {np.max(final_accounts):.2f}") 