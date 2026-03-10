"""
Generate LaTeX tables for the paper.
Table 1: Main results (final performance)
Table 2: Computational overhead (from actual timing data)
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['cheetah-run', 'walker-run', 'quadruped-run', 'humanoid-run', 'dog-run']

METHODS = {
    'tdmpc2': {'main': 'tdmpc2_{task}', 'validate': 'validate_baseline_{task}'},
    'tdmpc2_rnd': {'main': 'tdmpc2_rnd_{task}', 'validate': None},
    'freeguide': {'main': 'freeguide_{task}', 'validate': 'validate_freeguide_{task}'},
}

PARAM_COUNTS = {
    'tdmpc2': 5.4,
    'tdmpc2_rnd': 5.5,
    'freeguide': 7.9,
}


def get_final_performance(task, exp_name, seeds=range(1, 6)):
    """Get final episode reward mean ± std across seeds."""
    dfs = load_data(task, exp_name, seeds)
    if not dfs:
        return None, None, 0
    finals = [df['episode_reward'].iloc[-1] for df in dfs]
    return np.mean(finals), np.std(finals), len(dfs)


def get_elapsed_time(task, exp_name, seeds=range(1, 6)):
    """Get final elapsed_time (seconds) from train.csv across seeds.
    Returns mean elapsed time in hours, or None if no data."""
    times = []
    for seed in seeds:
        csv_path = LOGS_ROOT / task / str(seed) / exp_name / 'train.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'elapsed_time' in df.columns and len(df) > 0:
                times.append(df['elapsed_time'].iloc[-1] / 3600.0)
    if times:
        return np.mean(times), np.std(times), len(times)
    return None, None, 0


def main():
    # Table 1: Main results
    rows = []
    for task in TASKS:
        row = [TASK_LABELS.get(task, task)]
        best_mean = -float('inf')
        values = []
        for method, names in METHODS.items():
            mean, std, n = None, None, 0
            for key in ['main', 'validate']:
                name = names.get(key)
                if name:
                    mean, std, n = get_final_performance(task, name.format(task=task))
                    if mean is not None:
                        break
            if mean is not None:
                values.append((mean, std, n))
                if mean > best_mean:
                    best_mean = mean
            else:
                values.append((None, None, 0))

        for mean, std, n in values:
            if mean is not None:
                bold = mean >= best_mean - 0.01
                fmt = f'\\textbf{{{mean:.1f} $\\pm$ {std:.1f}}}' if bold else f'{mean:.1f} $\\pm$ {std:.1f}'
                row.append(fmt)
            else:
                row.append('--')
        rows.append(row)

    # Write Table 1
    with open(TABLES_DIR / 'table1.tex', 'w') as f:
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Final episode return (mean $\\pm$ std) on DMControl tasks at 3M steps.}\n')
        f.write('\\label{tab:main_results}\n')
        f.write('\\begin{tabular}{lccc}\n')
        f.write('\\toprule\n')
        f.write('Task & TD-MPC2 & TD-MPC2 + RND & FreeGuide \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            f.write(' & '.join(row) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'Saved: {TABLES_DIR}/table1.tex')

    # Table 2: Computational overhead — compute from actual data per task
    overhead_methods = ['tdmpc2', 'tdmpc2_rnd', 'freeguide']
    # Collect timing data: {method: {task: (mean_hours, std_hours, n)}}
    timing_data = {}
    for method in overhead_methods:
        timing_data[method] = {}
        names = METHODS[method]
        for task in TASKS:
            for key in ['main', 'validate']:
                name = names.get(key)
                if name:
                    mean_h, std_h, n = get_elapsed_time(task, name.format(task=task))
                    if mean_h is not None:
                        timing_data[method][task] = (mean_h, std_h, n)
                        break

    with open(TABLES_DIR / 'table2.tex', 'w') as f:
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Computational overhead: wall-clock time (hours) to reach 3M steps per task.}\n')
        f.write('\\label{tab:overhead}\n')

        # Check if we have per-task data
        has_per_task = any(timing_data[m] for m in overhead_methods)

        if has_per_task:
            # Per-task timing table
            task_cols = [t for t in TASKS if any(t in timing_data[m] for m in overhead_methods)]
            if not task_cols:
                task_cols = TASKS
            col_spec = 'lc' + 'c' * len(task_cols) + 'c'
            f.write(f'\\begin{{tabular}}{{{col_spec}}}\n')
            f.write('\\toprule\n')
            task_headers = ' & '.join(TASK_LABELS.get(t, t) for t in task_cols)
            f.write(f'Method & Params (M) & {task_headers} & Avg. Overhead \\\\\n')
            f.write('\\midrule\n')

            # Compute baseline average for overhead %
            bl_avgs = {}
            for task in task_cols:
                if task in timing_data['tdmpc2']:
                    bl_avgs[task] = timing_data['tdmpc2'][task][0]

            for method in overhead_methods:
                params = PARAM_COUNTS.get(method, '--')
                cells = []
                overhead_pcts = []
                for task in task_cols:
                    if task in timing_data[method]:
                        h, _, _ = timing_data[method][task]
                        cells.append(f'{h:.1f}')
                        if task in bl_avgs and method != 'tdmpc2' and bl_avgs[task] > 0:
                            overhead_pcts.append((h - bl_avgs[task]) / bl_avgs[task] * 100)
                    else:
                        cells.append('--')
                task_cells = ' & '.join(cells)
                if overhead_pcts:
                    avg_pct = np.mean(overhead_pcts)
                    overhead_str = f'+{avg_pct:.0f}\\%' if avg_pct > 0 else f'{avg_pct:.0f}\\%'
                elif method == 'tdmpc2':
                    overhead_str = '---'
                else:
                    overhead_str = '--'
                f.write(f'{METHOD_LABELS.get(method, method)} & {params} & {task_cells} & {overhead_str} \\\\\n')
        else:
            # Fallback: static table with placeholder values
            f.write('\\begin{tabular}{lccc}\n')
            f.write('\\toprule\n')
            f.write('Method & Params (M) & Time/1M steps (h) & Overhead \\\\\n')
            f.write('\\midrule\n')
            for method in overhead_methods:
                params = PARAM_COUNTS.get(method, '--')
                label = METHOD_LABELS.get(method, method)
                overhead = '---' if method == 'tdmpc2' else '--'
                f.write(f'{label} & {params} & -- & {overhead} \\\\\n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'Saved: {TABLES_DIR}/table2.tex')


if __name__ == '__main__':
    main()
