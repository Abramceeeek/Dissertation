import numpy as np
from rila.hedging import simulate_dynamic_hedge, analyze_hedging_performance
from rila.payoff import apply_rila_payoff

def test_simulate_dynamic_hedge_shape():
    # Simple GBM-like paths
    n_steps, n_paths = 10, 5
    S0 = 100
    paths = np.full((n_steps+1, n_paths), S0)
    for t in range(1, n_steps+1):
        paths[t] = paths[t-1] * 1.01  # deterministic up move
    r = 0.02
    q = 0.01
    sigma = 0.2
    buffer = 0.1
    cap = 0.15
    hedge_pnl, hedge_portfolio = simulate_dynamic_hedge(paths, S0, r, q, sigma, buffer, cap)
    assert hedge_pnl.shape == (n_paths,)
    assert hedge_portfolio.shape == (n_steps+1, n_paths)

def test_analyze_hedging_performance_risk_reduction():
    # Unhedged PnL: all -10, hedged PnL: all -5
    unhedged = np.full(100, -10.0)
    hedged = np.full(100, -5.0)
    stats = analyze_hedging_performance(hedged, unhedged)
    assert stats['risk_reduction_var95'] > 0
    assert stats['risk_reduction_std'] > 0 