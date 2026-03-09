"""
Generate LaTeX tables for the paper.
Table 1: Main results (final performance)
Table 2: Computational overhead
"""
import sys
sys.path.insert(0, '.')
from plot_config import *

TASKS = ['cheetah-run', 'walker-walk', 'walker-run', 'humanoid-walk',
         'humanoid-run', 'dog-walk', 'dog-run']

METHODS = {
    'tdmpc2': {'main': 'tdmpc2_{task}', 'validate': 'validate_baseline_{task}'},
    'freeguide_qev': {'main': 'freeguide_qev_{task}', 'validate': None},
    'freeguide': {'main': 'freeguide_{task}', 'validate': 'validate_freeguide_{task}'},
}


def get_final_performance(task, exp_name, seeds=range(1, 6)):
    """Get final episode reward mean ± std across seeds."""
    dfs = load_data(task, exp_name, seeds)
    if not dfs:
        return None, None, 0
    finals = [df['episode_reward'].iloc[-1] for df in dfs]
    return np.mean(finals), np.std(finals), len(dfs)


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
        f.write('Task & TD-MPC2 & FreeGuide-QEV & FreeGuide \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            f.write(' & '.join(row) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'Saved: {TABLES_DIR}/table1.tex')

    # Table 2: Computational overhead
    with open(TABLES_DIR / 'table2.tex', 'w') as f:
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Computational overhead of FreeGuide components.}\n')
        f.write('\\label{tab:overhead}\n')
        f.write('\\begin{tabular}{lcccc}\n')
        f.write('\\toprule\n')
        f.write('Method & Params (M) & Train Time (h) & Planning FPS & GPU Mem (GB) \\\\\n')
        f.write('\\midrule\n')
        f.write('TD-MPC2 & 5.0 & -- & -- & -- \\\\\n')
        f.write('FreeGuide (K=3) & 7.3 & -- & -- & -- \\\\\n')
        f.write('FreeGuide (K=5) & 8.9 & -- & -- & -- \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\vspace{-2mm}\n')
        f.write('\\end{table}\n')
    print(f'Saved: {TABLES_DIR}/table2.tex')


if __name__ == '__main__':
    main()
