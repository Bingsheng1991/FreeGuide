"""
Statistical significance tests: Welch t-test and Cohen's d.
"""
import sys
sys.path.insert(0, '.')
from plot_config import *
from scipy import stats

TASKS = ['cheetah-run', 'walker-walk', 'walker-run', 'humanoid-walk',
         'humanoid-run', 'dog-walk', 'dog-run']

METHODS = {
    'tdmpc2': {'main': 'tdmpc2_{task}', 'validate': 'validate_baseline_{task}'},
    'freeguide': {'main': 'freeguide_{task}', 'validate': 'validate_freeguide_{task}'},
}


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def get_finals(task, method_names, seeds=range(1, 6)):
    """Get final returns for a method across seeds."""
    for key in ['main', 'validate']:
        name = method_names.get(key)
        if name:
            dfs = load_data(task, name.format(task=task), seeds)
            if dfs:
                return [df['episode_reward'].iloc[-1] for df in dfs]
    return None


def main():
    rows = []
    for task in TASKS:
        bl_finals = get_finals(task, METHODS['tdmpc2'])
        fg_finals = get_finals(task, METHODS['freeguide'])

        if bl_finals is None or fg_finals is None or len(bl_finals) < 2 or len(fg_finals) < 2:
            rows.append((TASK_LABELS.get(task, task), '--', '--', '--', '--', '--'))
            continue

        t_stat, p_value = stats.ttest_ind(fg_finals, bl_finals, equal_var=False)
        d = cohens_d(fg_finals, bl_finals)
        sig = '*' if p_value < 0.05 else ''
        if p_value < 0.01:
            sig = '**'
        if p_value < 0.001:
            sig = '***'

        rows.append((
            TASK_LABELS.get(task, task),
            f'{np.mean(bl_finals):.1f}',
            f'{np.mean(fg_finals):.1f}',
            f'{t_stat:.2f}',
            f'{p_value:.4f}{sig}',
            f'{d:.2f}'
        ))

    with open(TABLES_DIR / 'stats.tex', 'w') as f:
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Statistical significance: Welch t-test and Cohen\'s d comparing FreeGuide vs TD-MPC2.}\n')
        f.write('\\label{tab:stats}\n')
        f.write('\\begin{tabular}{lccccc}\n')
        f.write('\\toprule\n')
        f.write('Task & TD-MPC2 & FreeGuide & $t$-stat & $p$-value & Cohen\'s $d$ \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            f.write(' & '.join(row) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'Saved: {TABLES_DIR}/stats.tex')


if __name__ == '__main__':
    main()
