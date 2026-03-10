"""
Figure 8: Information gain dynamics — 3 columns (info_gain, beta, ensemble_loss)
across all available tasks (one row per task).
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['cheetah-run', 'walker-run', 'quadruped-run', 'humanoid-run', 'dog-run']

# Task-specific line colors (distinguishable when overlaid)
TASK_COLORS = {
    'cheetah-run': '#1f77b4',
    'walker-run': '#ff7f0e',
    'quadruped-run': '#2ca02c',
    'humanoid-run': '#d62728',
    'dog-run': '#9467bd',
}


def load_train_csv(task, seed, exp_name):
    """Load training metrics from train.csv in the experiment log directory."""
    csv_path = LOGS_ROOT / task / str(seed) / exp_name / 'train.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        fg_cols = ['freeguide/beta', 'freeguide/info_gain_edd',
                   'freeguide/info_gain_qev', 'freeguide/ensemble_loss']
        if all(c in df.columns for c in fg_cols):
            df = df.dropna(subset=['freeguide/beta'])
            if len(df) > 0:
                return df
    return None


def collect_task_data(task):
    """Collect all available seed data for a task's FreeGuide experiment."""
    exp_names = [f'freeguide_{task}', f'validate_freeguide_{task}']
    all_data = []
    for exp_name in exp_names:
        for seed in range(1, 6):
            data = load_train_csv(task, seed, exp_name)
            if data is not None:
                all_data.append(data)
        if all_data:
            break
    return all_data


def main():
    # Collect data for all tasks
    task_data = {}
    for task in TASKS:
        data = collect_task_data(task)
        if data:
            task_data[task] = data

    if not task_data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data available yet',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        save_fig(fig, 'fig8_dynamics')
        return

    n_tasks = len(task_data)
    fig, axes = plt.subplots(n_tasks, 3, figsize=(15, 3.5 * n_tasks),
                             squeeze=False)

    for row_idx, (task, data_list) in enumerate(task_data.items()):
        color = TASK_COLORS.get(task, '#333333')
        label = TASK_LABELS.get(task, task)

        for i, data in enumerate(data_list):
            alpha = 0.8 if len(data_list) == 1 else max(0.3, 1.0 - 0.15 * i)
            seed_label = f'seed {i+1}' if i == 0 and len(data_list) > 1 else None

            # Column 0: Information Gain (EDD + QEV)
            axes[row_idx, 0].plot(data['step'], data['freeguide/info_gain_edd'],
                                  color=color, alpha=alpha, linewidth=1.2,
                                  label='EDD' if i == 0 else None)
            axes[row_idx, 0].plot(data['step'], data['freeguide/info_gain_qev'],
                                  color=color, alpha=alpha * 0.6, linewidth=1.0,
                                  linestyle='--', label='QEV' if i == 0 else None)

            # Column 1: Beta
            axes[row_idx, 1].plot(data['step'], data['freeguide/beta'],
                                  color=color, alpha=alpha, linewidth=1.2)

            # Column 2: Ensemble loss
            axes[row_idx, 2].plot(data['step'], data['freeguide/ensemble_loss'],
                                  color=color, alpha=alpha, linewidth=1.2)

        # Row labels and legends
        axes[row_idx, 0].set_ylabel(f'{label}\nInfo Gain')
        axes[row_idx, 0].legend(fontsize=8, loc='upper right')
        axes[row_idx, 1].set_ylabel(r'$\beta$')
        axes[row_idx, 2].set_ylabel('Ensemble Loss')

        for col in range(3):
            axes[row_idx, col].grid(True, alpha=0.3)

    # Column titles on top row
    col_titles = ['Information Gain Components', r'Adaptive $\beta$ Schedule', 'Ensemble Dynamics Loss']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)

    # X-axis labels on bottom row only
    for col in range(3):
        axes[-1, col].set_xlabel('Environment Steps')

    n_display = len(task_data)
    fig.suptitle(f'FreeGuide Training Dynamics ({n_display} tasks)',
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig8_dynamics')


if __name__ == '__main__':
    main()
