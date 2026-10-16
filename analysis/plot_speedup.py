"""
Fig. 3 and Supplementary Fig. SF2: without deliberation (delta = 0), median convergence time to consensus on A as a
function of the rewiring probability rho. Dots: medians over the runs at each rho, with 95% bootstrap confidence
intervals; lines: least-squares linear fits over all values of rho. The legend gives the relative reduction of the
fitted convergence time between rho = 0 and rho = 1; the 95% bootstrap interval of that reduction (resampling runs
within each rho) is printed.

Usage:  python plot_speedup.py <aggregated_sweep.csv> --pd 0.02 --pk 0.035 [--N 50 100] [--q 0.7 0.8 0.9] [--out f.pdf]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_aggregated, select

COLORS = {0.7: '#1f77b4', 0.8: '#2ca02c', 0.9: '#ff7f0e'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--pd', type=float, default=0.02)
    ap.add_argument('--pk', type=float, default=0.035)
    ap.add_argument('--N', type=int, nargs='+', default=[50, 100])
    ap.add_argument('--q', type=float, nargs='+', default=[0.7, 0.8, 0.9])
    ap.add_argument('--T', type=int, default=10000)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', default='fig_speedup.pdf')
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    df = load_aggregated(args.csv)
    fig, axes = plt.subplots(1, len(args.N), figsize=(3.6 * len(args.N), 3.2), squeeze=False)
    for ax, N in zip(axes[0], args.N):
        for q in args.q:
            sub = select(df, N=N, q=q, dp=args.pd, p=args.pk, mt=args.T, dt=0).sort_values('rw')
            if sub.empty:
                continue
            rhos = sub.rw.values.astype(float)
            data = list(sub.t.values)
            censored = sum(int((d >= args.T).sum()) for d in data)
            med = np.array([np.median(d) for d in data])
            boots = np.array([np.median(rng.choice(d, (args.boot, len(d)), replace=True), axis=1) for d in data])
            lo, hi = np.percentile(boots, 2.5, axis=1), np.percentile(boots, 97.5, axis=1)
            slope, icpt = np.polyfit(rhos, med, 1)
            red = -slope / icpt
            reds = []
            for b in range(args.boot):
                s_, i_ = np.polyfit(rhos, boots[:, b], 1)
                reds.append(-s_ / i_)
            r_lo, r_hi = np.percentile(reds, [2.5, 97.5])
            c = COLORS.get(q)
            ax.errorbar(rhos, med / 1e3, yerr=[(med - lo) / 1e3, (hi - med) / 1e3], fmt='o', ms=3, color=c, alpha=0.9,
                        lw=0.8, label=rf'$q_B={q}$')
            xx = np.linspace(0, 1, 50)
            ax.plot(xx, (icpt + slope * xx) / 1e3, color=c, lw=1.8)
            ax.text(1.07, (icpt + slope) / 1e3, f'{100 * red:.0f}% faster', color=c, fontsize=8, va='center')
            print(f'N={N} q_B={q}: reduction {100 * red:.1f}% [95% CI {100 * r_lo:.1f}, {100 * r_hi:.1f}], '
                  f'{len(rhos)} values of rho, censored runs: {censored}')
        ax.set_xlabel(r'Rewiring probability $\rho$', fontsize=10)
        ax.set_ylabel(r'Median convergence time ($\times10^3$)', fontsize=10)
        ax.set_xlim(-0.04, 1.38); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches='tight')
    print('saved', args.out)
