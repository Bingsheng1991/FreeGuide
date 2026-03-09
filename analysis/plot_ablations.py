"""
Figure 4: Ablation study — learning curves for ablation variants.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['walker-run', 'humanoid-walk']

ABLATION_METHODS = {
    'tdmpc2': 'ablation_tdmpc2_{task}',
    'qev_only': 'ablation_qev_only_{task}',
    'edd_only': 'ablation_edd_only_{task}',
    'fixed_beta_01': 'ablation_fixed_beta_01_{task}',
    'fixed_beta_03': 'ablation_fixed_beta_03_{task}',
    'fixed_beta_05': 'ablation_fixed_beta_05_{task}',
    'freeguide': 'ablation_freeguide_{task}',
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, task in enumerate(TASKS):
        ax = axes[i]
        methods = {m: n.format(task=task) for m, n in ABLATION_METHODS.items()}
        plot_learning_curve(ax, task, methods, seeds=range(1, 4))

    fig.suptitle('Ablation Study', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig4_ablations')


if __name__ == '__main__':
    main()
