"""
Extended-horizon simulations (Supplementary Text ST3).
  - Supplementary Fig. SF5: (a) median first-passage time to consensus on A over a (rho, delta) grid (log scale; the
    white contour marks the horizon T = 1e4 of the main sweep); (b) adaptation probability within the extended horizon.
  - Supplementary Table STab2: for selected combinations, adaptation probability within 1e4 and within the extended
    horizon, and the median first-passage time. Combinations split over several seeds are pooled.

Usage:  python plot_extended.py <aggregated_extended.csv> [--out figS_extended_grid.pdf]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common import load_aggregated, select

TABLE = [(50, 0.8, 0.02, 75, 0.9), (50, 0.8, 0.02, 100, 1.0), (50, 0.9, 0.02, 50, 1.0),
         (50, 0.7, 0.02, 100, 1.0), (100, 0.8, 0.02, 100, 1.0), (50, 0.8, 0.01, 75, 0.9)]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--grid', type=float, nargs=4, default=[50, 0.8, 0.02, 0.035], metavar=('N', 'q_B', 'p_d', 'p_k'))
    ap.add_argument('--T', type=int, default=1000000)
    ap.add_argument('--out', default='figS_extended_grid.pdf')
    args = ap.parse_args()
    df = load_aggregated(args.csv)

    print('Supplementary Table STab2')
    print('   N  q_B   p_d  delta  rho   runs  P(1e4)  P(T)     T        median first-passage time')
    for N, q, pd_, dt, rw in TABLE:
        rows = select(df, N=N, q=q, dp=pd_, dt=dt, rw=rw)
        for mt in sorted(rows.mt.unique()):
            t = np.concatenate(list(rows[rows.mt == mt].t.values))
            print(f'{N:4d} {q:4.1f} {pd_:5.2f} {dt:6d} {rw:4.1f} {len(t):6d} {np.mean(t < 1e4):7.2f} {np.mean(t < mt):5.2f} '
                  f'{mt:9.0e} {np.median(t):12.3g}')

    N, q, pd_, pk = args.grid
    sub = select(df, N=int(N), q=q, dp=pd_, p=pk, mt=args.T)
    # keep the regular grid: values of delta present for every rho (drops isolated off-grid combinations)
    rhos = sorted(sub.rw.unique())
    counts = sub.groupby('dt').rw.nunique()
    deltas = sorted(counts[counts == len(rhos)].index)
    sub = sub[sub.dt.isin(deltas)].drop_duplicates(subset=['dt', 'rw'])
    M = np.full((len(deltas), len(rhos)), np.nan); P = M.copy()
    for _, r in sub.iterrows():
        i, j = deltas.index(r['dt']), rhos.index(r['rw'])
        M[i, j] = np.median(r.t); P[i, j] = np.mean(r.t < args.T)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(wspace=0.35))
    dr, dd = rhos[1] - rhos[0], deltas[1] - deltas[0]
    ext = [rhos[0] - dr / 2, rhos[-1] + dr / 2, deltas[0] - dd / 2, deltas[-1] + dd / 2]
    ax = axes[0]
    im = ax.imshow(M, origin='lower', aspect='auto', cmap='magma', norm=LogNorm(vmin=1e3, vmax=1e6), extent=ext,
                   interpolation='nearest')
    R, D = np.meshgrid(rhos, deltas)
    cs = ax.contour(R, D, np.log10(M), levels=[4.0], colors='white', linewidths=2)
    ax.clabel(cs, fmt={4.0: r'$10^4$'}, fontsize=11, colors='white')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('median time to consensus on A (timesteps)', fontsize=11)
    ax.set_xlabel(r'rewiring probability $\rho$', fontsize=13); ax.set_ylabel(r'deliberation time $\delta$', fontsize=13)
    ax.text(-0.12, 1.03, 'a', transform=ax.transAxes, fontsize=18, fontweight='bold')
    ax = axes[1]
    im2 = ax.imshow(P, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1, extent=ext, interpolation='nearest')
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label(rf'adaptation probability within $T=10^{{{int(np.log10(args.T))}}}$', fontsize=11)
    ax.set_xlabel(r'rewiring probability $\rho$', fontsize=13); ax.set_ylabel(r'deliberation time $\delta$', fontsize=13)
    ax.text(-0.12, 1.03, 'b', transform=ax.transAxes, fontsize=18, fontweight='bold')
    fig.savefig(args.out, bbox_inches='tight', pad_inches=0.05)
    print('saved', args.out)
    print('median first-passage time (rows: delta, columns: rho)')
    for i, d in enumerate(deltas):
        print(f'{d:4d} ' + ' '.join(f'{M[i, j]:8.0f}' for j in range(len(rhos))))
