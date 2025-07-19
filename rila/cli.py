import argparse
import sys
from rila.models import simulate_gbm, simulate_heston, simulate_rough_vol
from rila.payoff import apply_rila_payoff
from rila.hedging import simulate_dynamic_hedge, analyze_hedging_performance
from rila.config import S0, n_paths, T, N, heston_params, buffer_level, cap_level
import numpy as np
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="RILA Risk & Hedging CLI")
    subparsers = parser.add_subparsers(dest='command')

    # Simulate
    sim_parser = subparsers.add_parser('simulate', help='Simulate price paths')
    sim_parser.add_argument('--model', choices=['gbm', 'heston', 'roughvol'], required=True)
    sim_parser.add_argument('--output', required=True, help='Output CSV file for paths')

    # Payoff
    payoff_parser = subparsers.add_parser('payoff', help='Apply RILA payoff to paths')
    payoff_parser.add_argument('--paths', required=True, help='CSV file with simulated paths')
    payoff_parser.add_argument('--output', required=True, help='Output CSV file for final accounts')

    # Hedge
    hedge_parser = subparsers.add_parser('hedge', help='Run dynamic hedging simulation')
    hedge_parser.add_argument('--paths', required=True, help='CSV file with simulated paths')
    hedge_parser.add_argument('--output', required=True, help='Output CSV file for hedging PnL')

    args = parser.parse_args()

    if args.command == 'simulate':
        if args.model == 'gbm':
            mu = 0.01
            sigma = 0.2
            S = simulate_gbm(S0, mu, sigma, T, N, n_paths)
        elif args.model == 'heston':
            mu = 0.01
            S = simulate_heston(S0, heston_params['v0'], mu, heston_params['kappa'], heston_params['theta'], heston_params['sigma_v'], heston_params['rho'], T, N, n_paths)
        elif args.model == 'roughvol':
            mu = 0.01
            S = simulate_rough_vol(S0, mu, 0.04, 1.5, 0.1, T, N, n_paths)
        else:
            print('Unknown model')
            sys.exit(1)
        pd.DataFrame(S).to_csv(args.output)
        print(f"✅ Simulated {args.model} paths saved to {args.output}")

    elif args.command == 'payoff':
        paths = pd.read_csv(args.paths, index_col=0)
        start_values = paths.iloc[0].values
        end_values = paths.iloc[-1].values
        returns = (end_values - start_values) / start_values
        returns = np.clip(returns, -0.9, 1.5)
        credited = apply_rila_payoff(returns, buffer_level, cap_level)
        final_accounts = S0 * (1 + credited)
        pd.Series(final_accounts).to_csv(args.output, header=['final_account'])
        print(f"✅ RILA payoff applied and results saved to {args.output}")

    elif args.command == 'hedge':
        paths = pd.read_csv(args.paths, index_col=0).values
        S0_local = paths[0, 0]
        r = 0.02
        q = 0.01
        sigma = 0.2
        hedge_pnl, _ = simulate_dynamic_hedge(paths, S0_local, r, q, sigma, buffer_level, cap_level)
        pd.Series(hedge_pnl).to_csv(args.output, header=['hedge_pnl'])
        print(f"✅ Hedging simulation complete. PnL saved to {args.output}")

    else:
        parser.print_help()

if __name__ == '__main__':
    main() 