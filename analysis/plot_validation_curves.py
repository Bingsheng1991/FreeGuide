"""
Validation curves: Compare baseline vs FreeGuide on walker-run and humanoid-run (200K steps).
Saves to logs/validation_curves.png
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

EXPERIMENTS = {
    'walker-run': {
        'tdmpc2': 'validate_baseline_walker-run',
        'freeguide': 'validate_freeguide_walker-run',
    },
    'humanoid-run': {
        'tdmpc2': 'validate_baseline_humanoid-run',
        'freeguide': 'validate_freeguide_humanoid-run',
    },
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (task, methods) in enumerate(EXPERIMENTS.items()):
        ax = axes[i]
        plot_learning_curve(ax, task, methods, seeds=[1], window=1)

    fig.suptitle('FreeGuide Validation (200K steps, seed=1)', fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Save to logs directory
    out_path = PROJECT_ROOT / 'logs' / 'validation_curves.png'
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
