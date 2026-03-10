"""
Figure 3: Sample efficiency — steps to reach 80% of baseline final performance.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['cheetah-run', 'walker-run', 'quadruped-run', 'humanoid-run', 'dog-run']

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
    """Return mean steps to reach threshold across seeds.
    If a seed never reaches the threshold, use MAX_STEPS for that seed.
    Returns (mean_steps, std_steps, n_reached, n_total).
    """
    steps_list = []
    for df in dfs:
        above = df[df[col] >= threshold]
        if len(above) > 0:
            steps_list.append(above['step'].iloc[0])
        else:
            steps_list.append(MAX_STEPS)
    n_reached = sum(1 for s in steps_list if s < MAX_STEPS)
    return np.mean(steps_list), np.std(steps_list), n_reached, len(steps_list)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    methods_to_plot = ['tdmpc2', 'tdmpc2_rnd', 'freeguide']
    tasks_with_data = []
    all_steps = {m: [] for m in methods_to_plot}
    all_stds = {m: [] for m in methods_to_plot}
    all_not_reached = {m: [] for m in methods_to_plot}  # True if some seeds never reached

    for task in TASKS:
        # Try to load baseline data (needed for threshold)
        bl_dfs = None
        for prefix, names in METHODS_NAMES.items():
            if 'tdmpc2' in names:
                bl_name = names['tdmpc2'].format(task=task)
                bl_dfs = load_data(task, bl_name)
                if bl_dfs:
                    break
        if not bl_dfs:
            continue

        # Compute threshold from baseline final performance
        bl_finals = [df['episode_reward'].iloc[-1] for df in bl_dfs]
        threshold = 0.8 * np.mean(bl_finals)

        has_any = False
        for method in methods_to_plot:
            dfs = None
            for prefix, names in METHODS_NAMES.items():
                if method in names:
                    exp_name = names[method].format(task=task)
                    dfs = load_data(task, exp_name)
                    if dfs:
                        break
            if dfs:
                has_any = True
                mean_s, std_s, n_reached, n_total = steps_to_threshold(dfs, threshold)
                all_steps[method].append(mean_s)
                all_stds[method].append(std_s)
                all_not_reached[method].append(n_reached < n_total)
            else:
                all_steps[method].append(None)
                all_stds[method].append(None)
                all_not_reached[method].append(False)

        if has_any:
            tasks_with_data.append(TASK_LABELS.get(task, task))

    if not tasks_with_data:
        print("No data available for sample efficiency plot.")
        fig.text(0.5, 0.5, 'No data available yet', ha='center', va='center', fontsize=14)
        save_fig(fig, 'fig3_efficiency')
        return

    x = np.arange(len(tasks_with_data))
    n_methods = len(methods_to_plot)
    width = 0.8 / n_methods

    for i, method in enumerate(methods_to_plot):
        vals = []
        errs = []
        for j in range(len(tasks_with_data)):
            v = all_steps[method][j]
            s = all_stds[method][j]
            vals.append((v if v is not None else 0) / 1000)
            errs.append((s if s is not None else 0) / 1000)

        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      yerr=errs, label=METHOD_LABELS.get(method, method),
                      color=COLORS.get(method, '#333333'), capsize=3)

        # Mark bars where some seeds never reached threshold
        for j, bar in enumerate(bars):
            if all_steps[method][j] is not None and all_not_reached[method][j]:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errs[j] + 20,
                        '*', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Steps to 80% Baseline (×1000)')
    ax.set_title('Sample Efficiency: Steps to Reach 80% of Baseline Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_with_data, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add footnote about * marker
    ax.annotate('* = some seeds did not reach threshold (counted as 3M steps)',
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=8,
                fontstyle='italic', color='gray')

    fig.tight_layout()
    save_fig(fig, 'fig3_efficiency')


if __name__ == '__main__':
    main()
