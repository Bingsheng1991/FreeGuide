"""
Figure 3: Sample efficiency — steps to reach 80% of baseline final performance.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['cheetah-run', 'walker-run', 'quadruped-run', 'humanoid-run', 'dog-run']

METHODS_MAP = {
    'tdmpc2': {'tdmpc2': 'tdmpc2_{task}', 'freeguide': 'freeguide_{task}'},
    'validate': {'tdmpc2': 'validate_baseline_{task}', 'freeguide': 'validate_freeguide_{task}'},
}


def steps_to_threshold(dfs, threshold, col='episode_reward'):
    """Return mean steps to reach threshold across seeds."""
    steps_list = []
    for df in dfs:
        above = df[df[col] >= threshold]
        if len(above) > 0:
            steps_list.append(above['step'].iloc[0])
    if steps_list:
        return np.mean(steps_list), np.std(steps_list)
    return None, None


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    tasks_with_data = []
    baseline_steps_list = []
    freeguide_steps_list = []
    baseline_stds = []
    freeguide_stds = []

    for task in TASKS:
        # Try main experiment names, fallback to validation
        for prefix, methods in METHODS_MAP.items():
            bl_name = methods['tdmpc2'].format(task=task)
            fg_name = methods['freeguide'].format(task=task)
            bl_dfs = load_data(task, bl_name)
            fg_dfs = load_data(task, fg_name)
            if bl_dfs and fg_dfs:
                break
        else:
            continue

        # Get baseline final performance
        bl_finals = [df['episode_reward'].iloc[-1] for df in bl_dfs]
        threshold = 0.8 * np.mean(bl_finals)

        bl_steps, bl_std = steps_to_threshold(bl_dfs, threshold)
        fg_steps, fg_std = steps_to_threshold(fg_dfs, threshold)

        if bl_steps is not None and fg_steps is not None:
            tasks_with_data.append(TASK_LABELS.get(task, task))
            baseline_steps_list.append(bl_steps)
            freeguide_steps_list.append(fg_steps)
            baseline_stds.append(bl_std if bl_std else 0)
            freeguide_stds.append(fg_std if fg_std else 0)

    if not tasks_with_data:
        print("No data available for sample efficiency plot.")
        fig.text(0.5, 0.5, 'No data available yet', ha='center', va='center', fontsize=14)
        save_fig(fig, 'fig3_efficiency')
        return

    x = np.arange(len(tasks_with_data))
    width = 0.35

    ax.bar(x - width/2, np.array(baseline_steps_list) / 1000, width,
           yerr=np.array(baseline_stds) / 1000, label='TD-MPC2',
           color=COLORS['tdmpc2'], capsize=3)
    ax.bar(x + width/2, np.array(freeguide_steps_list) / 1000, width,
           yerr=np.array(freeguide_stds) / 1000, label='FreeGuide',
           color=COLORS['freeguide'], capsize=3)

    ax.set_ylabel('Steps to 80% Baseline (×1000)')
    ax.set_title('Sample Efficiency: Steps to Reach 80% of Baseline Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_with_data, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_fig(fig, 'fig3_efficiency')


if __name__ == '__main__':
    main()
