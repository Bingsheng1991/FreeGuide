"""
Figure 6: Information gain dynamics — 3-panel plot showing info_gain, beta, ensemble_loss over training.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *


def load_train_log(task, seed, exp_name):
    """Try to load training metrics from the log directory."""
    log_dir = LOGS_ROOT / task / str(seed) / exp_name
    # Check for eval.csv (we log freeguide metrics in train logs)
    # Since we may not have dedicated FreeGuide metric CSVs,
    # we'll parse from the experiment log files
    log_file = PROJECT_ROOT / 'logs' / f'{exp_name}_seed{seed}.log'
    if not log_file.exists():
        return None

    steps, ig_edd, ig_qev, betas, ens_losses = [], [], [], [], []
    with open(log_file) as f:
        for line in f:
            # Parse freeguide metrics from log output
            if 'freeguide/beta' in line:
                # This would need proper parsing based on actual log format
                pass

    if not steps:
        return None
    return pd.DataFrame({
        'step': steps, 'info_gain_edd': ig_edd,
        'info_gain_qev': ig_qev, 'beta': betas,
        'ensemble_loss': ens_losses
    })


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    task = 'humanoid-run'
    exp_name = 'freeguide_humanoid-run'

    # Try to load data
    has_data = False
    for seed in range(1, 6):
        data = load_train_log(task, seed, exp_name)
        if data is not None:
            has_data = True
            break

    if not has_data:
        # Also try validation experiments
        exp_name = 'validate_freeguide_humanoid-run'
        for seed in [1]:
            data = load_train_log(task, seed, exp_name)
            if data is not None:
                has_data = True
                break

    if not has_data:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data available yet',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        axes[0].plot(data['step'], data['info_gain_edd'], color=COLORS['freeguide'], label='EDD')
        axes[0].plot(data['step'], data['info_gain_qev'], color=COLORS['tdmpc2_rnd'], label='QEV')
        axes[0].set_ylabel('Information Gain')
        axes[0].legend()

        axes[1].plot(data['step'], data['beta'], color=COLORS['freeguide'])
        axes[1].set_ylabel(r'$\beta$')

        axes[2].plot(data['step'], data['ensemble_loss'], color=COLORS['edd_only'])
        axes[2].set_ylabel('Ensemble Loss')

    titles = ['Information Gain Components', r'Adaptive $\beta$ Schedule', 'Ensemble Dynamics Loss']
    for ax, title in zip(axes, titles):
        ax.set_xlabel('Environment Steps')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'FreeGuide Training Dynamics ({TASK_LABELS.get(task, task)})',
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig6_dynamics')


if __name__ == '__main__':
    main()
