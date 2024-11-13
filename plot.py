import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np

# Set the style for seaborn
sns.set(style="whitegrid")

def load_data():
    # Read all CSV files matching the pattern
    files = glob.glob('*_results_S*.csv')
    df_list = []

    for file in files:
        # Extract size and strategy from filename
        basename = os.path.basename(file)
        match = re.match(r'(\d+x\d+)_results_(S\d+).csv', basename)
        if match:
            size = match.group(1)
            strategy = match.group(2)
        else:
            continue  # Skip files that don't match the pattern
        
        # Read the CSV file
        df = pd.read_csv(file)
        # Add size and strategy columns
        df['size'] = size
        df['strategy'] = strategy
        # Append to the list
        df_list.append(df)

    # Concatenate all data into a single DataFrame
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

def plot_total_time_boxplot(df, size):
    df_size = df[df['size'] == size]
    plt.figure(figsize=(10,6))
    sns.boxplot(x='strategy', y='total_time', data=df_size)
    plt.title(f'Comparison of Total Solving Time for Different Strategies ({size})')
    plt.xlabel('Strategy')
    plt.ylabel('Total Time (s)')
    plt.savefig(f'total_time_comparison_{size}.png')
    plt.show()

def plot_cdf_total_time(df, size):
    df_size = df[df['size'] == size]
    plt.figure(figsize=(10,6))
    for strategy in df_size['strategy'].unique():
        data = df_size[df_size['strategy'] == strategy]['total_time'].sort_values()
        yvals = np.arange(len(data))/float(len(data))
        plt.plot(data, yvals, label=strategy)
    plt.title(f'Cumulative Distribution of Total Solving Time ({size})')
    plt.xlabel('Total Time (s)')
    plt.ylabel('Proportion of Puzzles Solved')
    plt.legend(title='Strategy')
    plt.savefig(f'cdf_total_time_{size}.png')
    plt.show()

def plot_performance_profile(df, size):
    df_size = df[df['size'] == size]
    strategies = df_size['strategy'].unique()
    plt.figure(figsize=(10,6))

    # For each strategy, compute the performance ratio
    pivot_table = df_size.pivot_table(values='total_time', index='file', columns='strategy')
    min_times = pivot_table.min(axis=1)
    perf_ratio = pivot_table.divide(min_times, axis=0)
    tau_values = np.linspace(1, perf_ratio.max().max(), 100)

    for strategy in strategies:
        counts = [np.mean(perf_ratio[strategy] <= tau) for tau in tau_values]
        plt.plot(tau_values, counts, label=strategy)

    plt.title(f'Performance Profile of Strategies ({size})')
    plt.xlabel('Performance Ratio (τ)')
    plt.ylabel('Proportion of Puzzles Solved')
    plt.legend(title='Strategy')
    plt.savefig(f'performance_profile_{size}.png')
    plt.show()

def plot_metrics_comparison(df, size):
    df_size = df[df['size'] == size]
    metrics = ['calls', 'unit_props', 'literal_choices', 'max_depth', 'backtracks', 'conflicts', 'heuristic_time']
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.boxplot(x='strategy', y=metric, data=df_size)
        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Strategies ({size})')
        plt.xlabel('Strategy')
        plt.ylabel(metric.replace("_", " ").title())
        plt.savefig(f'{metric}_comparison_{size}.png')
        plt.show()

def main():
    df_all = load_data()

    # Get list of sizes
    sizes = df_all['size'].unique()

    # Generate plots for each size
    for size in sizes:
        plot_total_time_boxplot(df_all, size)
        plot_cdf_total_time(df_all, size)
        plot_performance_profile(df_all, size)
        plot_metrics_comparison(df_all, size)

if __name__ == "__main__":
    main()
