"""
Figure 8: Information gain dynamics — 3-panel plot showing info_gain, beta, ensemble_loss over training.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *


def load_train_csv(task, seed, exp_name):
    """Load training metrics from train.csv in the experiment log directory."""
    csv_path = LOGS_ROOT / task / str(seed) / exp_name / 'train.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Filter to rows that have FreeGuide metrics (non-NaN)
        fg_cols = ['freeguide/beta', 'freeguide/info_gain_edd',
                   'freeguide/info_gain_qev', 'freeguide/ensemble_loss']
        if all(c in df.columns for c in fg_cols):
            df = df.dropna(subset=['freeguide/beta'])
            if len(df) > 0:
                return df
    return None


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    task = 'humanoid-run'
    exp_names = ['freeguide_humanoid-run', 'validate_freeguide_humanoid-run']

    # Collect data across seeds
    all_data = []
    for exp_name in exp_names:
        for seed in range(1, 6):
            data = load_train_csv(task, seed, exp_name)
            if data is not None:
                all_data.append(data)
        if all_data:
            break

    if not all_data:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data available yet',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        for i, data in enumerate(all_data):
            alpha = 1.0 if i == 0 else 0.3
            axes[0].plot(data['step'], data['freeguide/info_gain_edd'],
                        color=COLORS['freeguide'], label='EDD' if i == 0 else None, alpha=alpha)
            axes[0].plot(data['step'], data['freeguide/info_gain_qev'],
                        color=COLORS['tdmpc2_rnd'], label='QEV' if i == 0 else None, alpha=alpha)
            axes[1].plot(data['step'], data['freeguide/beta'],
                        color=COLORS['freeguide'], alpha=alpha)
            axes[2].plot(data['step'], data['freeguide/ensemble_loss'],
                        color=COLORS['edd_only'], alpha=alpha)

        axes[0].set_ylabel('Information Gain')
        axes[0].legend()
        axes[1].set_ylabel(r'$\beta$')
        axes[2].set_ylabel('Ensemble Loss')

    titles = ['Information Gain Components', r'Adaptive $\beta$ Schedule', 'Ensemble Dynamics Loss']
    for ax, title in zip(axes, titles):
        ax.set_xlabel('Environment Steps')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'FreeGuide Training Dynamics ({TASK_LABELS.get(task, task)})',
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig8_dynamics')


if __name__ == '__main__':
    main()
