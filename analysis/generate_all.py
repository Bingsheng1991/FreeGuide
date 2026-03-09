"""
Generate all figures and tables for the FreeGuide paper.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    'plot_main_results.py',
    'plot_sample_efficiency.py',
    'plot_ablations.py',
    'plot_ensemble_k.py',
    'plot_info_dynamics.py',
    'compute_tables.py',
    'statistical_tests.py',
]

def main():
    script_dir = Path(__file__).parent
    failed = []
    for script in SCRIPTS:
        script_path = script_dir / script
        print(f'\n{"="*60}')
        print(f'Running: {script}')
        print(f'{"="*60}')
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_dir),
                capture_output=True, text=True, timeout=120
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f'STDERR: {result.stderr}')
                failed.append(script)
        except Exception as e:
            print(f'ERROR: {e}')
            failed.append(script)

    print(f'\n{"="*60}')
    if failed:
        print(f'Failed scripts: {failed}')
    else:
        print('All scripts completed successfully!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
