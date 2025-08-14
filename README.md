# The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk Management Approach

This repository contains the complete implementation of an MSc dissertation project analyzing Solvency II capital requirements for RILA (Registered Index-Linked Annuity) products under stochastic volatility models with dynamic hedging strategies.

## Research Question

**How do dynamic option hedging strategies and stochastic volatility models affect the regulatory capital charges for equity-linked variable annuity products under Solvency II?**

## Project Structure

### Core Implementation
- **Code/rila_payoff.py**: RILA payoff functions and static replication
- **Code/12_dynamic_hedging.py**: Dynamic hedging engine (delta/vega hedging)
- **Code/13_scr_one_year.py**: Solvency II one-year SCR calculation
- **Code/curves.py**: Interest rate and dividend yield curve management
- **Code/14_run_experiments.py**: Experiment runner (models × frequencies)

### Supporting Modules
- **Code/heston_pricing_*.py**: Heston model implementation and Carr-Madan pricing
- **Code/Step*.py**: Data processing, calibration, and simulation scripts
- **tests/**: Unit tests for RILA replication and SCR metrics
- **Data/**: Market data (SPX options, yield curves, dividend rates)

## How This Answers the Research Question

### 1. Solvency II SCR Framework Implementation

The project implements the **change in Own Funds (ΔOF) methodology** for one-year SCR calculation:

- **Own Funds (OF)** = Assets - Best Estimate Liability (BEL)
- **ΔOF** = OF₁ᵧ - OF₀ (change over one year)
- **SCR** = VaR 99.5% and CTE 99.5% of (-ΔOF)

This directly measures the regulatory capital requirement under Solvency II Article 101, where the SCR ensures the insurer can absorb losses occurring once in 200 years (99.5% confidence).

### 2. Stochastic Volatility Model Impact

The study compares **Heston** and **Rough Volatility** models against each other:

- **Heston Model**: Mean-reverting stochastic volatility with closed-form option pricing via Carr-Madan FFT
- **Rough Volatility**: Fractional Brownian motion-driven volatility (uses Heston proxy for option pricing)
- **No GBM baseline**: Focus purely on advanced stochastic volatility effects

The models generate different volatility clustering and tail behavior, directly affecting:
- BEL valuations at t=0 and t=1
- SCR distributions and quantiles
- Hedge effectiveness under different volatility regimes

### 3. Dynamic Hedging Strategy Effects

The dynamic hedging engine tests three rebalancing frequencies:
- **Daily**: Frequent rebalancing, low hedge error, high transaction costs
- **Weekly**: Balanced approach
- **Monthly**: Infrequent rebalancing, higher hedge error, low transaction costs

**Impact on SCR**:
- **Hedged SCR** = VaR₉₉.₅%(-(Asset₁ - BEL₁) + (Asset₀ - BEL₀))
- **Unhedged SCR** = VaR₉₉.₅%(-(0 - BEL₁) + (0 - BEL₀)) = VaR₉₉.₅%(BEL₀ - BEL₁)

Effective hedging reduces SCR by stabilizing Own Funds changes, with the optimal frequency balancing hedge error reduction against transaction costs.

### 4. Enterprise Risk Management Approach

The project adopts an **Enterprise Risk Management perspective** by:

- **Holistic Risk View**: Analyzing combined market, liability, and operational (hedging) risks
- **Capital Optimization**: Measuring how hedging strategies reduce regulatory capital requirements
- **Model Risk Assessment**: Comparing capital charges across different stochastic volatility models
- **Cost-Benefit Analysis**: Trading off hedging costs against capital relief

## Quick Start

### Prerequisites
```bash
pip install numpy pandas scipy matplotlib seaborn pytest
```

### Data Preparation
```bash
python Code/Step02_Data_Cleaning_and_Processing.py
python Code/riskfree_clean_prepare.py
```

### Model Calibration
```bash
python Code/Step08_Heston_Calibration_Differential_Evolution.py --date 2021-06-01 --save-params params/heston_20210601.json
```

### Path Simulation
```bash
python Code/Step10_Heston_Simulation.py --out data/paths_heston_1y.npz --params params/heston_20210601.json --seed 42
python Code/Step11_Rough_Volatility_Simulation.py --out data/paths_rough_1y.npz --seed 42
```

### Run Complete Experiment Grid
```bash
python Code/14_run_experiments.py \
  --models heston roughvol \
  --rebalance daily weekly monthly \
  --n_paths 50000 --tc_bps 1.0 --seed 42
```

This generates:
- **results/scr_summary.csv**: SCR metrics by model and frequency
- **results/plots/**: Comparison charts and ΔOF distributions
- **results/experiment_config.json**: Reproducible configuration

## Key Results Interpretation

The **results/scr_summary.csv** contains:
- **VaR_99.5, CTE_99.5**: Solvency II capital requirements
- **mean_dOF, stdev_dOF**: Own Funds change statistics
- **mean_hedge_error**: Hedging effectiveness measure
- **TC_bps**: Transaction cost analysis

**Higher SCR** indicates:
- More volatile liability valuations
- Less effective hedging
- Greater capital needs for regulatory compliance

**Lower SCR** indicates:
- Better hedging performance
- More stable Own Funds evolution
- Capital efficiency improvements

## Testing

```bash
# Test RILA replication accuracy
pytest tests/test_rila_replication.py -v

# Test SCR calculation correctness  
pytest tests/test_scr_metrics.py -v

# Run all tests
pytest tests/ -v
```

## Academic Context

This implementation directly addresses the research gap in **quantitative risk management for insurance products** by:

1. **Methodological Contribution**: First implementation of Solvency II ΔOF methodology for RILA products under rough volatility
2. **Practical Application**: Demonstrable impact of model choice and hedging frequency on regulatory capital
3. **Industry Relevance**: Provides insurers with quantitative tools for capital optimization and model validation

The results inform both **regulatory capital management** and **product design decisions** in the insurance industry, bridging academic stochastic volatility research with practical risk management applications.

## Visualization

- Use `rila.plotting` for CDF plots, capital bar charts, and worst-case path visualizations.
- All main scripts and the notebook include example plots.

## Advanced Features

- **Annual Reset RILA**: Use `apply_rila_annual_reset` in `rila.payoff` for annual reset logic with buffer, cap, fee, and participation.
- **Scenario Toggles**: Change parameters in `rila/config.py` to experiment with different product designs.
- **Dynamic Hedging**: Use `rila.hedging` for robust, modular hedging simulation and risk analysis.

## Troubleshooting

- If you encounter import errors, ensure your working directory is the project root and PYTHONPATH includes the root.
- For large simulations, ensure you have sufficient memory (vectorized code is efficient but can be memory-intensive).
- For plotting issues, ensure `matplotlib` is installed and working.

## Requirements
- Python 3.8+
- numpy, pandas, matplotlib, scipy, pytest

## Notes
- See `Chatgpt instructions.txt` for a detailed project roadmap and best practices.
- Each script is self-contained and prints progress/status.
- The codebase is modular and ready for extension, testing, and publication.

---