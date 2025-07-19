import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(data, label=None, color='blue', xlabel='Value', title='CDF', save_path=None):
    data = np.sort(data)
    cdf = np.arange(1, len(data)+1) / len(data)
    plt.figure(figsize=(8, 5))
    plt.plot(data, cdf, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_capital_bar_chart(capital_dict, title='Capital Requirement Comparison', save_path=None):
    labels = list(capital_dict.keys())
    values = [capital_dict[k] for k in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color='skyblue', edgecolor='black')
    plt.ylabel('Capital Requirement')
    plt.title(title)
    plt.grid(axis='y')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_worst_case_path(paths, final_accounts, n=1, title='Worst-Case Paths', save_path=None):
    # Find indices of n worst final accounts
    idx = np.argsort(final_accounts)[:n]
    plt.figure(figsize=(10, 6))
    for i in idx:
        plt.plot(paths[:, i], label=f'Path {i+1} (Final: {final_accounts[i]:.2f})')
    plt.xlabel('Time Step')
    plt.ylabel('Account Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show() 