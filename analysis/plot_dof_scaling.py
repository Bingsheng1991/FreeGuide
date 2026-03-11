"""
Figure 4: DoF scaling — sample efficiency improvement vs action dimensionality.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['cheetah-run', 'walker-run', 'quadruped-run', 'humanoid-run', 'dog-run']
DOF = [6, 6, 12, 21, 38]

MAX_STEPS = 3_000_000

METHODS_NAMES = {
    'main': {
        'tdmpc2': 'tdmpc2_{task}',
        'tdmpc2_rnd': 'tdmpc2_rnd_{task}',
        'freeguide': 'freeguide_{task}',
    },
    'validate': {
        'tdmpc2': 'validate_baseline_{task}',
        'freeguide': 'validate_freeguide_{task}',
    },
}


def steps_to_threshold(dfs, threshold, col='episode_reward'):
    """Return mean steps to reach threshold across seeds."""
    steps_list = []
    for df in dfs:
        above = df[df[col] >= threshold]
        if len(above) > 0:
            steps_list.append(above['step'].iloc[0])
        else:
            steps_list.append(MAX_STEPS)
    return np.mean(steps_list)


def load_method_data(task, method):
    """Try loading data for a method with fallback exp_names."""
    for prefix, names in METHODS_NAMES.items():
        if method in names:
            exp_name = names[method].format(task=task)
            dfs = load_data(task, exp_name)
            if dfs:
                return dfs
    return None


def main():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    improvements = {'freeguide': [], 'tdmpc2_rnd': []}
    valid_dofs = []

    for task, dof in zip(TASKS, DOF):
        # Load baseline data for threshold
        bl_dfs = load_method_data(task, 'tdmpc2')
        if not bl_dfs:
            for m in improvements:
                improvements[m].append(None)
            continue

        bl_finals = [df['episode_reward'].iloc[-1] for df in bl_dfs]
        threshold = 0.8 * np.mean(bl_finals)
        bl_steps = steps_to_threshold(bl_dfs, threshold)

        has_data = False
        for method in ['freeguide', 'tdmpc2_rnd']:
            dfs = load_method_data(task, method)
            if dfs:
                method_steps = steps_to_threshold(dfs, threshold)
                # Improvement = (baseline_steps - method_steps) / baseline_steps * 100
                if bl_steps > 0:
                    improvement = (bl_steps - method_steps) / bl_steps * 100
                else:
                    improvement = 0.0
                improvements[method].append(improvement)
                has_data = True
            else:
                improvements[method].append(None)

        if has_data:
            valid_dofs.append(dof)

    # Plot
    for method, color, marker in [
        ('freeguide', COLORS['freeguide'], 'o'),
        ('tdmpc2_rnd', COLORS['tdmpc2_rnd'], 's'),
    ]:
        x_vals, y_vals = [], []
        for i, (dof, imp) in enumerate(zip(DOF, improvements[method])):
            if imp is not None:
                x_vals.append(dof)
                y_vals.append(imp)

        if x_vals:
            ax.plot(x_vals, y_vals, color=color, marker=marker, linewidth=2,
                    markersize=8, label=METHOD_LABELS.get(method, method))

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Action Dimensionality (DoF)')
    ax.set_ylabel('Sample Efficiency Improvement (%)')
    ax.set_title('DoF Scaling: Sample Efficiency Improvement vs. Task Complexity')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add task labels at data points
    for i, (task, dof) in enumerate(zip(TASKS, DOF)):
        if improvements['freeguide'][i] is not None:
            ax.annotate(TASK_LABELS.get(task, task),
                        xy=(dof, improvements['freeguide'][i]),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=7, color='gray')

    fig.tight_layout()
    save_fig(fig, 'fig4_dof_scaling')


if __name__ == '__main__':
    main()
