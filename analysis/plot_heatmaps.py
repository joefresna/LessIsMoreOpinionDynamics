"""
Fig. 5 and Supplementary Figs. SF1 and SF3: adaptation probability (fraction of runs reaching consensus on A within
T) over the (rho, delta) plane. Rows: population sizes; columns: quality of the inferior option. The white contour
marks P = 0.9. Also prints, for each panel, the deliberation time at which P crosses 0.5 at rho = 1 (linear
interpolation between grid points), used for the comparison with the mean-field prediction.

Usage:  python plot_heatmaps.py <aggregated_sweep.csv> --pd 0.02 --pk 0.035 [--N 50 100] [--q 0.7 0.8 0.9] [--out f.pdf]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_aggregated, select


def delta_half(rows):
    """Deliberation time at which the adaptation probability first drops below 0.5 (linear interpolation)."""
    rows = rows.sort_values('dt')
    d, P = rows.dt.values.astype(float), rows.P.values
    for i in range(len(d) - 1):
        if P[i] >= 0.5 > P[i + 1]:
            return d[i] + (P[i] - 0.5) / (P[i] - P[i + 1]) * (d[i + 1] - d[i])
    return np.nan


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--pd', type=float, default=0.02)
    ap.add_argument('--pk', type=float, default=0.035)
    ap.add_argument('--N', type=int, nargs='+', default=[50, 100])
    ap.add_argument('--q', type=float, nargs='+', default=[0.7, 0.8, 0.9])
    ap.add_argument('--T', type=int, default=10000)
    ap.add_argument('--out', default='fig_heatmaps.pdf')
    args = ap.parse_args()

    df = load_aggregated(args.csv)
    nr, nc = len(args.N), len(args.q)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc + 1.0, 2.9 * nr), squeeze=False, sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.12, hspace=0.18))
    im = None
    for i, N in enumerate(args.N):
        for j, q in enumerate(args.q):
            ax = axes[i, j]
            sub = select(df, N=N, q=q, dp=args.pd, p=args.pk, mt=args.T)
            if sub.empty:
                ax.set_axis_off(); continue
            piv = sub.pivot_table(index='dt', columns='rw', values='P').sort_index()
            rhos, dts = piv.columns.values.astype(float), piv.index.values.astype(float)
            im = ax.imshow(piv.values, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1,
                           extent=[rhos[0], rhos[-1], dts[0], dts[-1]], interpolation='bilinear')
            R, D = np.meshgrid(rhos, dts)
            cs = ax.contour(R, D, piv.values, levels=[0.9], colors='white', linewidths=1.6)
            ax.clabel(cs, fmt={0.9: 'P=0.9'}, fontsize=7, colors='white')
            if i == 0:
                ax.set_title(rf'$q_B = {q}$', fontsize=12)
            if i == nr - 1:
                ax.set_xlabel(r'Rewiring probability $\rho$', fontsize=11)
            if j == 0:
                ax.set_ylabel(r'Deliberation time $\delta$', fontsize=11)
            ax.text(0.97, 0.95, rf'$N={N}$', transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
            ax.text(-0.04, 1.02, 'abcdefghij'[i * nc + j], transform=ax.transAxes, fontsize=13, fontweight='bold',
                    ha='right', va='bottom')
            at1 = sub[np.isclose(sub.rw, 1.0)]
            print(f'N={N} q_B={q}: P(rho=0, delta=max)={piv.iloc[-1, 0]:.2f}; delta at which P=0.5 for rho=1: {delta_half(at1):.1f}')
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cb.set_label('Adaptation probability', fontsize=11)
    fig.savefig(args.out, bbox_inches='tight', pad_inches=0.03)
    print('saved', args.out)
