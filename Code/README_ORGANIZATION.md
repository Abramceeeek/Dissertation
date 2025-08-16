# Code Folder Organization Guide

## Overview
This folder contains all the Python code files organized in sequential steps from data preparation to final analysis. All files are now properly numbered and organized in logical order.

## File Organization by Steps

### **Phase 1: Configuration & Data Preparation (Steps 00-04)**
- **Step00_Configuration.py** - Main configuration file with all parameters
- **Step01_Risk_Free_Rate_Trimming.py** - Filter interest rate data
- **Step02_Risk_Free_Rate_Processing.py** - Clean and process interest rate data
- **Step03_Data_Merging_and_Preparation.py** - Merge multiple years of SPX option data
- **Step04_Data_Cleaning_and_Processing.py** - Clean and process option data

### **Phase 2: Data Snapshot Creation (Steps 05-07)**
- **Step05_Create_Snapshot.py** - Create individual data snapshots
- **Step06_Target_IV_Surface_Visualization.py** - Visualize IV surface for specific dates
- **Step07_Create_All_Snapshots.py** - Create snapshots for all available dates

### **Phase 3: Implied Volatility Analysis (Steps 08-10)**
- **Step08_IV_Smile_Analysis.py** - Analyze IV smile patterns
- **Step09_IV_Surface_Construction.py** - Construct 3D IV surfaces
- **Step10_Complete_IV_Surface_Analysis.py** - Comprehensive IV surface analysis

### **Phase 4: Model Calibration (Steps 11-12)**
- **Step11_Heston_Model_Calibration.py** - Basic Heston model calibration
- **Step12_Heston_Calibration_Differential_Evolution.py** - Advanced Heston calibration using DE

### **Phase 5: Simulation Models (Steps 13-15)**
- **Step13_GBM_Simulation.py** - Geometric Brownian Motion simulation
- **Step14_Heston_Simulation.py** - Heston stochastic volatility simulation
- **Step15_Rough_Volatility_Simulation.py** - Rough volatility model simulation

### **Phase 6: GMAB Product Analysis (Steps 16-18)**
- **Step16_GMAB_under_GBM.py** - GMAB analysis under GBM model
- **Step17_GMAB_under_Heston.py** - GMAB analysis under Heston model
- **Step18_GMAB_under_RoughVol.py** - GMAB analysis under Rough Volatility model

### **Phase 7: Results & Analysis (Steps 19-20)**
- **Step19_Results_Summary_Table.py** - Generate summary tables and LaTeX output
- **Step20_Model_Calibration_Validation.py** - Validate model calibration results

### **Phase 8: Variable Annuity Framework (Steps 21-25)**
- **Step21_Variable_Annuity_Simulation_and_Hedging.py** - Main VA simulation and hedging
- **Step22_Black_Scholes_Utils.py** - Black-Scholes pricing utilities
- **Step23_Heston_Carr_Madan_Pricing.py** - Heston pricing using Carr-Madan method
- **Step24_Heston_Pricing_Utilities.py** - Heston characteristic function and pricing
- **Step25_VA_Configuration.py** - Variable Annuity configuration parameters

### **Phase 9: Utilities & Support (Steps 26-27)**
- **Step26_VA_Utilities.py** - Variable Annuity utility functions
- **Step27_Variable_Annuity_Utils.py** - Additional VA utility functions

### **Phase 10: Testing & Validation (Steps 28-29)**
- **Step28_Hedging_Test_Suite.py** - Test dynamic hedging strategies
- **Step29_Draft_Analysis.py** - Draft analysis and data exploration

### **Phase 11: Advanced Risk Management (Steps 30-32)**
- **Step30_Interest_Rate_and_Dividend_Curves.py** - Yield curve management
- **Step31_Dynamic_Hedging_Engine.py** - Dynamic hedging implementation
- **Step32_Solvency_II_SCR_Calculation.py** - Solvency II capital requirements

### **Phase 12: Core Product Module (Step 33)**
- **Step33_GMAB_Product_Module.py** - Core GMAB product implementation

### **Phase 13: Testing & Quality Assurance (Steps 34-36)**
- **Step34_Test_Models.py** - Test simulation models
- **Step35_Test_Payoff.py** - Test payoff calculations
- **Step36_Test_SCR_Metrics.py** - Test SCR calculation functions

## Key Benefits of This Organization

1. **Logical Flow**: Files follow the natural progression from data preparation to final analysis
2. **Clear Dependencies**: Each step builds upon previous steps
3. **Easy Navigation**: Sequential numbering makes it easy to find specific functionality
4. **Maintenance**: Clear organization makes debugging and updates easier
5. **Documentation**: Each phase has a clear purpose and scope

## Usage Notes

- **Start with Step00_Configuration.py** to set up all parameters
- **Follow the sequence** from Phase 1 to Phase 13 for complete analysis
- **Each phase can be run independently** once previous phases are completed
- **Output files** are generated in the Output/ directory
- **Data files** are expected in the Data/ directory

## File Dependencies

- **Steps 00-04**: Must be run first to prepare data
- **Steps 05-10**: Depend on cleaned data from previous steps
- **Steps 11-12**: Depend on data snapshots from previous steps
- **Steps 13-15**: Can be run independently for simulation testing
- **Steps 16-18**: Depend on simulation models and GMAB module
- **Steps 19+**: Depend on results from previous analysis steps

## Maintenance

- Keep the sequential numbering when adding new files
- Update this README when adding or removing files
- Ensure all import paths are updated when reorganizing
- Test the complete pipeline after any reorganization 