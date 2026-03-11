"""
Figure 6: Ablation study — learning curves for ablation variants.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['walker-run', 'humanoid-run']

ABLATION_METHODS = {
    'tdmpc2': ['tdmpc2_{task}', 'ablation_tdmpc2_{task}', 'validate_baseline_{task}'],
    'qev_only': ['ablation_qev_only_{task}'],
    'edd_only': ['ablation_edd_only_{task}'],
    'fixed_beta_01': ['ablation_fixed_beta_01_{task}', 'ablation_fixed_beta01_{task}'],
    'fixed_beta_03': ['ablation_fixed_beta_03_{task}', 'ablation_fixed_beta03_{task}'],
    'fixed_beta_05': ['ablation_fixed_beta_05_{task}', 'ablation_fixed_beta05_{task}'],
    'freeguide': ['freeguide_{task}', 'ablation_freeguide_{task}', 'validate_freeguide_{task}'],
}


def find_exp_name(task, candidates, seeds=range(1, 4)):
    """Try candidate exp_names in order, return the first one with data."""
    for name_tpl in candidates:
        name = name_tpl.format(task=task)
        dfs = load_data(task, name, seeds)
        if dfs:
            return name
    return candidates[0].format(task=task)  # fallback to first


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, task in enumerate(TASKS):
        ax = axes[i]
        methods = {m: find_exp_name(task, candidates) for m, candidates in ABLATION_METHODS.items()}
        plot_learning_curve(ax, task, methods, seeds=range(1, 4))

    fig.suptitle('Ablation Study', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig6_ablations')


if __name__ == '__main__':
    main()
