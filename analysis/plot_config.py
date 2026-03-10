"""
Shared plotting configuration for FreeGuide paper figures.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Global matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color scheme
COLORS = {
    'tdmpc2': '#4C72B0',
    'tdmpc2_rnd': '#8172B3',
    'freeguide': '#C44E52',
    'qev_only': '#DD8452',
    'edd_only': '#55A868',
    'fixed_beta_03': '#937860',
}

METHOD_LABELS = {
    'tdmpc2': 'TD-MPC2',
    'tdmpc2_rnd': 'TD-MPC2 + RND',
    'freeguide': 'FreeGuide',
    'qev_only': 'QEV Only',
    'edd_only': 'EDD Only',
    'fixed_beta_03': r'Fixed $\beta$=0.3',
}

TASK_LABELS = {
    'cheetah-run': 'Cheetah Run',
    'walker-run': 'Walker Run',
    'quadruped-run': 'Quadruped Run',
    'humanoid-run': 'Humanoid Run',
    'dog-run': 'Dog Run',
}

# Paths
PROJECT_ROOT = Path('/home/miller/Desktop/FreeGuide')
TDMPC2_ROOT = PROJECT_ROOT / 'tdmpc2' / 'tdmpc2'
LOGS_ROOT = TDMPC2_ROOT / 'logs'
FIGURES_DIR = PROJECT_ROOT / 'paper' / 'figures'
TABLES_DIR = PROJECT_ROOT / 'paper' / 'tables'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_csv(task, seed, exp_name):
    """Load eval.csv for a given experiment."""
    csv_path = LOGS_ROOT / task / str(seed) / exp_name / 'eval.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None


def load_data(task, method_exp_name, seeds=range(1, 6)):
    """Load data for a method across seeds. Returns list of DataFrames."""
    dfs = []
    for seed in seeds:
        df = load_eval_csv(task, seed, method_exp_name)
        if df is not None:
            dfs.append(df)
    return dfs


def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def compute_ci(dfs, col='episode_reward', window=5):
    """
    Compute mean and 95% CI from multiple seed DataFrames.
    Returns (steps, mean, lower, upper, n_seeds).
    """
    if not dfs:
        return None, None, None, None, 0

    # Find common step range
    min_len = min(len(df) for df in dfs)
    steps = dfs[0]['step'].values[:min_len]

    # Collect rewards
    all_rewards = []
    for df in dfs:
        rewards = df[col].values[:min_len]
        smoothed = smooth(rewards, window)
        all_rewards.append(smoothed)

    # Align lengths after smoothing
    min_smooth_len = min(len(r) for r in all_rewards)
    all_rewards = np.array([r[:min_smooth_len] for r in all_rewards])
    steps_smooth = steps[:min_smooth_len]

    mean = all_rewards.mean(axis=0)
    std = all_rewards.std(axis=0)
    n = len(dfs)
    ci = 1.96 * std / np.sqrt(max(n, 1))

    return steps_smooth, mean, mean - ci, mean + ci, n


def save_fig(fig, name, formats=('pdf', 'png')):
    """Save figure in multiple formats."""
    for fmt in formats:
        path = FIGURES_DIR / f'{name}.{fmt}'
        fig.savefig(path, format=fmt)
        print(f'Saved: {path}')
    plt.close(fig)


def plot_learning_curve(ax, task, methods, seeds=range(1, 6), window=5):
    """Plot learning curves for multiple methods on a single task."""
    for method, exp_name in methods.items():
        dfs = load_data(task, exp_name, seeds)
        steps, mean, lower, upper, n = compute_ci(dfs, window=window)
        if steps is None:
            continue
        color = COLORS.get(method, '#333333')
        label = f'{METHOD_LABELS.get(method, method)} (n={n})'
        ax.plot(steps, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(steps, lower, upper, alpha=0.2, color=color)

    ax.set_title(TASK_LABELS.get(task, task))
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.legend(loc='lower right', framealpha=0.8)
    ax.grid(True, alpha=0.3)
