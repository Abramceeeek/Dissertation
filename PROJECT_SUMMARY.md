# The Effects of Stochastic Volatility Models and Dynamic Hedging Strategies on Capital Requirements for Equity-Linked Variable Annuities: An Enterprise Risk Management Approach

**Author:** Abdurakhmonbek Fayzullaev  
**Degree:** MSc Actuarial Science and Data Analytics  
**Institution:** Queen Mary University of London  
**Supervisor:** Dr. Sutton

## Project Overview

This dissertation examines how different stochastic volatility models and dynamic hedging strategies affect capital requirements for equity-linked variable annuities within an enterprise risk management framework. The research provides practical insights for insurance companies managing variable annuity portfolios under modern regulatory frameworks like Solvency II.

## Research Questions

1. **Model Dependency in Capital Calculations**: How do different stochastic volatility models affect capital requirement calculations for variable annuity guarantees?

2. **Dynamic Hedging Effectiveness**: What is the quantitative effectiveness of dynamic hedging strategies in reducing capital requirements across different models?

3. **Enterprise Risk Management Implementation**: How should insurance companies integrate model choice and hedging strategies into their risk management frameworks?

## Project Structure

### Code Files (Step-by-Step Analysis)

1. **Step01_Data_Merging_and_Preparation.py** - Merges S&P 500 option data from multiple years (2018-2023)
2. **Step02_Data_Cleaning_and_Processing.py** - Cleans and prepares market data for analysis
3. **Step03_Target_IV_Surface_Visualization.py** - Creates implied volatility surface visualizations
4. **Step04_IV_Smile_Analysis.py** - Analyzes implied volatility smile characteristics
5. **Step05_IV_Surface_Construction.py** - Constructs comprehensive implied volatility surfaces
6. **Step06_Complete_IV_Surface_Analysis.py** - Performs complete volatility surface analysis
7. **Step07_Heston_Model_Calibration.py** - Calibrates Heston stochastic volatility model
8. **Step08_Heston_Calibration_Differential_Evolution.py** - Alternative calibration using differential evolution
9. **Step09_GBM_Simulation.py** - Simulates equity paths under Geometric Brownian Motion
10. **Step10_Heston_Simulation.py** - Simulates equity paths under Heston model
11. **Step11_Rough_Volatility_Simulation.py** - Simulates equity paths under rough volatility model
12. **Step12_Results_Summary_Table.py** - Creates comprehensive results summary tables
13. **Step13_Model_Calibration_Validation.py** - Validates model calibration results
14. **Step14_Variable_Annuity_Simulation_and_Hedging.py** - Main comprehensive analysis script

### Utility Files

- **VA_Configuration.py** - Central configuration file for Variable Annuity parameters
- **VA_Utilities.py** - General utility functions for Variable Annuity analysis
- **Variable_Annuity_Utils.py** - Specialized Variable Annuity calculation functions
- **Black_Scholes_Utils.py** - Black-Scholes option pricing utilities
- **Heston_Carr_Madan_Pricing.py** - Heston model pricing via Carr-Madan method
- **Heston_Pricing_Utilities.py** - General Heston model pricing functions
- **Risk_Free_Rate_Processing.py** - Risk-free rate data processing
- **Risk_Free_Rate_Trimming.py** - Risk-free rate data cleaning

### Key Data Sources

- **SPX Option Chain Data** (2018-2023): Market option prices for model calibration
- **SPX Index Prices** (2018-2023): Underlying equity index data
- **Risk-Free Yield Curves** (2018-2023): Treasury rates for pricing
- **Dividend Yield Data** (2018-2023): S&P 500 dividend yields

### Output Files

- **Simulation Results**: Monte Carlo simulation paths for different models
- **Risk Metrics**: VaR, Expected Shortfall, and other risk measures
- **Hedging Analysis**: Dynamic hedging effectiveness results
- **Model Comparisons**: Comparative analysis across stochastic volatility models
- **Capital Requirements**: Solvency II-inspired capital calculations

## Key Findings

The research demonstrates that:

1. **Model Choice Matters**: The Heston model requires 15-25% higher capital than simplified GBM models due to volatility clustering effects.

2. **Hedging is Highly Effective**: Dynamic hedging reduces capital requirements by 35-50% across all models while transforming risk profiles from negative to positive expected outcomes.

3. **Implementation Complexity**: More sophisticated models require substantially more computational resources but provide more realistic risk assessments.

4. **Regulatory Implications**: Model choice has material implications for regulatory capital calculations under frameworks like Solvency II.

## Practical Applications

This research provides actionable guidance for:

- **Insurance Companies**: Model selection for variable annuity risk management
- **Risk Managers**: Quantifying capital benefits of hedging programs
- **Regulators**: Understanding model risk in insurance capital calculations
- **Actuaries**: Implementing advanced stochastic volatility models in practice

## Technical Implementation

The project uses:

- **Python** for all numerical analysis and simulation
- **NumPy/SciPy** for mathematical computations
- **Pandas** for data manipulation
- **Matplotlib** for visualization
- **Monte Carlo Methods** for risk simulation
- **Optimization Algorithms** for model calibration

## Running the Analysis

1. Execute data preparation steps (Step01-Step06)
2. Run model calibration (Step07-Step08)
3. Perform simulations (Step09-Step11)
4. Generate results and analysis (Step12-Step14)

The main comprehensive analysis is in **Step14_Variable_Annuity_Simulation_and_Hedging.py**.

## Academic Contributions

- First comprehensive comparison of capital requirements across multiple stochastic volatility models for variable annuities
- Quantitative evidence on trade-offs between model sophistication and practical implementation
- Practical framework for integrating advanced models into enterprise risk management

## Repository Organization

The project has been reorganized from RILA-focused analysis to Variable Annuity research, with:

- Clear step-based file naming for easy workflow understanding
- Comprehensive documentation and humanized code comments
- Focused scope on capital requirements and enterprise risk management
- Removal of unnecessary data files and RILA-specific code

---

*This project represents comprehensive research into variable annuity risk management, providing both academic insights and practical guidance for insurance industry professionals.*