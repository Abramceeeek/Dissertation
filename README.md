# RILA Risk & Hedging Analysis Project

This project models, prices, and analyzes the risk of Registered Index-Linked Annuity (RILA) products under various equity models (GBM, Heston, Rough Volatility). It includes calibration, simulation, payoff logic, dynamic hedging, and risk/capital analysis.

## Project Structure

- **rila/**: Modular package with models, payoff, hedging, plotting, and config
- **Code/**: Example scripts and legacy code
- **tests/**: Unit tests for all core modules
- **Output/**: All results (plots, CSVs)

## Usage

### 1. Run from CLI

```bash
python -m rila.cli simulate --model heston --output Output/simulations/SPX_Heston_paths.csv
python -m rila.cli payoff --paths Output/simulations/SPX_Heston_paths.csv --output Output/heston_final_accounts.csv
python -m rila.cli hedge --paths Output/simulations/SPX_Heston_paths.csv --output Output/heston_hedge_pnl.csv
```

### 2. Run Example Scripts

- See `Code/8_simulate_spx_heston.py`, `Code/9_rila_heston.py`, etc. for script-based workflow.

### 3. Jupyter Notebook

- Open `RILA_Workflow_Demo.ipynb` for a full, interactive workflow demonstration.

### 4. Run Tests

```bash
pytest tests/
```

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