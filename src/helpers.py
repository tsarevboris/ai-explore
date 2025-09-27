import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_linear_data(n_samples=100, noise=1.0, seed=42):
    np.random.seed(seed)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + noise * np.random.randn(n_samples, 1)
    return X, y

def prepare_plot_data(X, y, title="Data"):
    plt.scatter(X, y, color='blue', s=10, alpha=0.5, label='Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)

def analyze_missing_data(df):
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    
    missing_table = pd.DataFrame({
        'Count': missing_data,
        'Percentage': missing_percent
    })
    return missing_table[missing_table['Count'] > 0].sort_values('Count', ascending=False)
