"""
Figure 5: Reward prediction vs planning modification.
(a) Reward prediction loss over training for TD-MPC2, +RND, FreeGuide.
(b) Latent state coverage heatmap at 500K steps (PCA 2D projection).
"""
import sys
sys.path.insert(0, '.')
from plot_config import *
from sklearn.decomposition import PCA

TASK = 'humanoid-run'
SEEDS = range(1, 6)

METHODS = {
    'tdmpc2': ['tdmpc2_{task}', 'validate_baseline_{task}'],
    'tdmpc2_rnd': ['tdmpc2_rnd_{task}', 'validate_rnd_{task}'],
    'freeguide': ['freeguide_{task}', 'validate_freeguide_{task}'],
}


def load_train_csv(task, seed, exp_name):
    """Load train.csv for a given experiment."""
    csv_path = LOGS_ROOT / task / str(seed) / exp_name / 'train.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None


def load_train_data(task, method_candidates, seeds):
    """Load train data trying candidate exp_names in order."""
    for name_tpl in method_candidates:
        name = name_tpl.format(task=task)
        dfs = []
        for seed in seeds:
            df = load_train_csv(task, seed, name)
            if df is not None:
                dfs.append(df)
        if dfs:
            return dfs, name
    return [], None


def load_latent_states(task, seed, exp_name, step=500000):
    """Load latent states from npz file."""
    npz_path = LOGS_ROOT / task / str(seed) / exp_name / 'latent_states' / f'latent_{step}.npz'
    if npz_path.exists():
        data = np.load(npz_path)
        return data['z']
    return None


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Panel (a): Reward prediction loss ===
    ax = axes[0]

    for method, candidates in METHODS.items():
        dfs, exp_name = load_train_data(TASK, candidates, SEEDS)
        if not dfs or 'reward_loss' not in dfs[0].columns:
            continue

        # Filter rows with valid reward_loss
        valid_dfs = []
        for df in dfs:
            df_valid = df.dropna(subset=['reward_loss'])
            if len(df_valid) > 0:
                valid_dfs.append(df_valid)

        if not valid_dfs:
            continue

        # Align to common length
        min_len = min(len(df) for df in valid_dfs)
        steps = valid_dfs[0]['step'].values[:min_len]
        all_losses = np.array([df['reward_loss'].values[:min_len] for df in valid_dfs])

        mean_loss = all_losses.mean(axis=0)
        std_loss = all_losses.std(axis=0)
        n = len(valid_dfs)
        ci = 1.96 * std_loss / np.sqrt(max(n, 1))

        # Smooth
        win = max(1, len(mean_loss) // 50)
        if win > 1:
            kernel = np.ones(win) / win
            mean_s = np.convolve(mean_loss, kernel, mode='valid')
            ci_s = np.convolve(ci, kernel, mode='valid')
            steps_s = steps[:len(mean_s)]
        else:
            mean_s, ci_s, steps_s = mean_loss, ci, steps

        color = COLORS.get(method, '#333333')
        label = f'{METHOD_LABELS.get(method, method)} (n={n})'
        ax.plot(steps_s, mean_s, color=color, label=label, linewidth=1.5)
        ax.fill_between(steps_s, mean_s - ci_s, mean_s + ci_s, alpha=0.2, color=color)

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Reward Prediction Loss')
    ax.set_title(f'(a) Reward Prediction Accuracy ({TASK_LABELS.get(TASK, TASK)})')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # === Panel (b): Latent state coverage at 500K ===
    ax = axes[1]

    all_z = {}
    for method, candidates in {'tdmpc2': METHODS['tdmpc2'], 'freeguide': METHODS['freeguide']}.items():
        z_list = []
        for name_tpl in candidates:
            name = name_tpl.format(task=TASK)
            for seed in SEEDS:
                z = load_latent_states(TASK, seed, name, step=500000)
                if z is not None:
                    z_list.append(z)
            if z_list:
                break
        if z_list:
            all_z[method] = np.concatenate(z_list, axis=0)

    if len(all_z) >= 2:
        # Fit PCA on combined data
        combined = np.concatenate(list(all_z.values()), axis=0)
        pca = PCA(n_components=2)
        pca.fit(combined)

        for method, z in all_z.items():
            z_2d = pca.transform(z)
            color = COLORS.get(method, '#333333')
            label = METHOD_LABELS.get(method, method)
            ax.scatter(z_2d[:, 0], z_2d[:, 1], c=color, alpha=0.3, s=5, label=label)

        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(f'(b) Latent State Coverage at 500K Steps')
        ax.legend(loc='upper right', framealpha=0.8, markerscale=5)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Latent state data not yet available\n(requires latent_500000.npz)',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.set_title(f'(b) Latent State Coverage at 500K Steps')

    fig.tight_layout()
    save_fig(fig, 'fig5_reward_vs_planning')


if __name__ == '__main__':
    main()
