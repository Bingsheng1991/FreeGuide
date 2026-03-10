"""
Figure 5: Ensemble K sensitivity — bar chart of final performance.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASK = 'walker-run'
K_VALUES = [2, 3, 5]
SEEDS = range(1, 4)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    means = []
    stds = []
    labels = []

    # Baseline (try P1 name first, then ablation prefix)
    bl_dfs = load_data(TASK, f'tdmpc2_{TASK}', SEEDS)
    if not bl_dfs:
        bl_dfs = load_data(TASK, f'ablation_tdmpc2_{TASK}', SEEDS)
    if not bl_dfs:
        bl_dfs = load_data(TASK, f'validate_baseline_{TASK}', SEEDS)
    if bl_dfs:
        finals = [df['episode_reward'].iloc[-1] for df in bl_dfs]
        means.append(np.mean(finals))
        stds.append(np.std(finals))
        labels.append(f'TD-MPC2\n(no ensemble)')
    else:
        means.append(0)
        stds.append(0)
        labels.append('TD-MPC2\n(no data)')

    for K in K_VALUES:
        # K=3 is the default, reuse P1 freeguide experiment
        if K == 3:
            candidates = [f'freeguide_{TASK}', f'ablation_ensemble_K3_{TASK}',
                          f'validate_freeguide_{TASK}']
        else:
            candidates = [f'ablation_ensemble_K{K}_{TASK}',
                          f'ensemble_K{K}_{TASK}']
        dfs = None
        for name in candidates:
            dfs = load_data(TASK, name, SEEDS)
            if dfs:
                break
        if dfs:
            finals = [df['episode_reward'].iloc[-1] for df in dfs]
            means.append(np.mean(finals))
            stds.append(np.std(finals))
            labels.append(f'K={K}')
        else:
            means.append(0)
            stds.append(0)
            labels.append(f'K={K}\n(no data)')

    x = np.arange(len(labels))
    colors = [COLORS['tdmpc2']] + [COLORS['freeguide']] * len(K_VALUES)
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Final Episode Return')
    ax.set_title(f'Ensemble Size Sensitivity ({TASK_LABELS[TASK]})')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_fig(fig, 'fig5_ensemble_k')


if __name__ == '__main__':
    main()
