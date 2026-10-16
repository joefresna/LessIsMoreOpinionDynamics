"""
Supplementary Fig. SF4: run-averaged trajectories of the agent-based model in three regimes (slow rewiring and long
deliberation; fast rewiring and long deliberation; fast rewiring and short deliberation), from the averages produced
by aggregate_evo.py. Runs that reached consensus before the horizon are held in the absorbing all-A state.

Usage:  python plot_trajectories.py <evo_mean_dir> [--N 50] [--out figS_abm_trajectories.pdf]
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#5B9BD5', '#ED7D31', '#70AD47', '#FF0000']
LABELS = ['$x_A$', '$d_A$', '$x_B$', '$d_B$']
REGIMES = [(0.1, 75, 'a', 'slow rewiring, long deliberation'), (0.9, 75, 'b', 'fast rewiring, long deliberation'),
           (0.9, 2, 'c', 'fast rewiring, short deliberation')]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--N', type=int, default=50)
    ap.add_argument('--T', type=int, default=30000)
    ap.add_argument('--out', default='figS_abm_trajectories.pdf')
    args = ap.parse_args()
    N = args.N
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(wspace=0.28))
    for ax, (rho, dt, lab, title) in zip(axes, REGIMES):
        f = os.path.join(args.src, f'LIME_N-{N}_mt-{args.T}_q-0.8_dp-0.01_net-erdos-renyi_p-0.035_dt-{dt}_rw-{rho}_mean.csv')
        d = pd.read_csv(f)
        ax.stackplot(d.t, d.xA, d.dA, d.xB, d.dB, colors=COLORS, labels=LABELS, alpha=0.92)
        ax.set_xlim(0, args.T); ax.set_ylim(0, 1)
        ax.set_xlabel('time (timesteps)', fontsize=12); ax.tick_params(labelsize=10)
        ax.set_title(rf'$\rho={rho}$, $\delta={dt}$ ({title})', fontsize=10.5, pad=30)
        ax.text(-0.1, 1.08, lab, transform=ax.transAxes, fontsize=17, fontweight='bold')
        ax2 = ax.secondary_xaxis('top', functions=(lambda t: t / N, lambda u: u * N))
        ax2.set_xlabel(rf'time (units of $N={N}$ timesteps)', fontsize=9); ax2.tick_params(labelsize=9)
    axes[0].set_ylabel('fraction of population', fontsize=12)
    axes[2].legend(loc='center right', fontsize=10, framealpha=0.9)
    fig.savefig(args.out, bbox_inches='tight', pad_inches=0.03)
    print('saved', args.out)
