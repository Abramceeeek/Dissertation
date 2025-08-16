import pandas as pd

hedge_df = pd.read_csv('Output/comprehensive_hedging_analysis.csv')

calib_local = pd.read_csv('Output/heston_calibrated_params_2018-06-01.csv')
calib_de = pd.read_csv('Output/heston_calibrated_params_2018-06-01_DE.csv')

print('--- Heston Calibration Parameters (Local) ---')
print(calib_local.to_string(index=False))
print('\n--- Heston Calibration Parameters (DE) ---')
print(calib_de.to_string(index=False))
print('\n--- Comprehensive Hedging Analysis ---')
print(hedge_df.to_string(index=False))

with open('all_results.txt', 'w') as f:
    f.write('--- Heston Calibration Parameters (Local) ---\n')
    calib_local.to_string(f, index=False)
    f.write('\n\n--- Heston Calibration Parameters (DE) ---\n')
    calib_de.to_string(f, index=False)
    f.write('\n\n--- Comprehensive Hedging Analysis ---\n')
    hedge_df.to_string(f, index=False)

with open('hedging_results_table.tex', 'w') as f:
    f.write(hedge_df.to_latex(index=False, caption='Comprehensive Hedging Results by Model and Strategy', label='tab:comprehensive_results'))

print('\nResults saved to all_results.txt and hedging_results_table.tex')
