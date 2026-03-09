"""
Figure 2: Main results — 7 tasks learning curves (2x4 grid).
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['cheetah-run', 'walker-walk', 'walker-run', 'humanoid-walk',
         'humanoid-run', 'dog-walk', 'dog-run']

METHODS = {
    'tdmpc2': 'tdmpc2_{task}',
    'freeguide_qev': 'freeguide_qev_{task}',
    'freeguide': 'freeguide_{task}',
}

# Also check validation experiment names
METHODS_VALIDATE = {
    'tdmpc2': 'validate_baseline_{task}',
    'freeguide': 'validate_freeguide_{task}',
}


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes[i]
        methods = {m: n.format(task=task) for m, n in METHODS.items()}
        # Also try validation names as fallback
        methods_val = {m: n.format(task=task) for m, n in METHODS_VALIDATE.items()}

        has_data = False
        for method, exp_name in methods.items():
            dfs = load_data(task, exp_name)
            if dfs:
                has_data = True
                break

        if not has_data:
            # Try validation experiment names
            methods = methods_val

        plot_learning_curve(ax, task, methods)

    # Hide empty subplot
    axes[-1].set_visible(False)

    fig.suptitle('FreeGuide vs TD-MPC2: DMControl Benchmark', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig2_main')


if __name__ == '__main__':
    main()
