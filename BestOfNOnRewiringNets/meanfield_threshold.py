"""
Structure of the mean-field threshold (Supplementary Text ST4, Supplementary Fig. SF6, Supplementary Table STab1).

  table   critical contact number E_c = (1 - p_d) delta_c at m = 1 for the parameters of the simulations, the critical
          deliberation time in timesteps N delta_c, and the limit p_d -> 0, where E_c tends to q_A/q_B - 1
          (linear-stability threshold of the all-B state); also checks that threshold by direct integration.
  figure  figS_meanfield_threshold.pdf:
          (a) fixed points versus delta (q_B = 0.8, p_d = 0.01, m = 1): consensus, trapped state, saddle;
          (b) leading eigenvalue at the trapped state, which vanishes at delta_c (saddle-node bifurcation);
          (c) trapping region in the (pi, E) plane, pi = p_d/s, E = s delta, for q_B = 0.7, 0.8, 0.9.
Usage:  python meanfield_threshold.py [table|figure|all] [--out-dir DIR] [--quick]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from meanfield_model import (Q_A, rhs, asymptotic, delta_c_bisection, critical_contacts, fixed_points, fold)


def delta_max_bisection(m, q_b, p_d, d_lo, d_hi=3000.0, rtol=1e-3):
    """Upper end of the trapping window: trapped at d_lo (just above delta_c), consensus at d_hi (bisection on log delta)."""
    trapped = lambda d: asymptotic(d, m, q_b, p_d)[0] < 0.5
    if trapped(d_hi):
        return np.inf
    lo, hi = d_lo, d_hi
    while hi / lo > 1 + rtol:
        mid = np.sqrt(lo * hi)
        if trapped(mid):
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def pi_max(q_b, m=1.0, tol=2e-4):
    """Largest discovery-to-contact ratio for which a trapped state exists for some delta (tested on a log grid)."""
    def has_trapping(p_d):
        return any(asymptotic(d, m, q_b, p_d)[0] < 0.5 for d in np.geomspace(0.2, 60, 40))
    lo, hi = 0.01, 0.08
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if has_trapping(mid):
            lo = mid
        else:
            hi = mid
    p_d = 0.5 * (lo + hi)
    return p_d / (1 - p_d)


def table():
    print('Critical values at m = 1 (Supplementary Table STab1)')
    print(' p_d   q_B    E_c   q_A/q_B-1   N*delta_c (N=50)   N*delta_c (N=100)')
    for p_d in [0.01, 0.02]:
        for q_b in [0.7, 0.8, 0.9]:
            Ec = critical_contacts(q_b, p_d)
            dc = Ec / (1 - p_d)
            print(f'{p_d:5.2f} {q_b:5.1f} {Ec:6.3f} {Q_A / q_b - 1:9.3f} {50 * dc:12.0f} {100 * dc:18.0f}')
    print('Limit of rare discovery: E_c(p_d) at m = 1')
    for q_b in [0.7, 0.8, 0.9]:
        vals = ', '.join(f'p_d={p}: {critical_contacts(q_b, p):.3f}' for p in [1e-4, 1e-3, 1e-2, 2e-2])
        print(f'  q_B={q_b}: {vals}   (limit q_A/q_B - 1 = {Q_A / q_b - 1:.3f})')
    print('Linear-stability check at p_d = 0 (s = 1): growth of a small amount of A just below/above the threshold')
    for q_b in [0.7, 0.8, 0.9]:
        E_th = Q_A / q_b - 1
        for E in [0.9 * E_th, 1.1 * E_th]:
            lam = 1.0 / E
            J = np.array([[-1.0, lam], [Q_A / q_b, -(lam + 1.0)]])
            sol = solve_ivp(rhs, [0, 2000], [1e-4, 0, 1 - 1e-4, 0], args=(lam, 1.0, q_b, 0.0),
                            method='LSODA', rtol=1e-10, atol=1e-13)
            print(f'  q_B={q_b}, E={E:.3f}: leading eigenvalue {np.linalg.eigvals(J).real.max():+.4f}, '
                  f'x_A at t=2000: {sol.y[0, -1]:.2e}')


def main(out_dir, quick=False):
    QUICK = quick
    qB, pd, m = 0.8, 0.01, 1.0
    # ---- (a),(b): branches versus delta
    deltas = np.linspace(0.3, 2.0, 35 if QUICK else 120)
    branches = {'trapped': [], 'saddle': [], 'consensus': []}
    for d in deltas:
        fps = fixed_points(d, m, qB, pd)
        small = [w for w in fps if w[0][0] < 0.9]
        cons = [w for w in fps if w[0][0] >= 0.9]
        stable_small = [w for w in small if w[1] < 0]
        unstable_small = [w for w in small if w[1] >= 0]
        branches['trapped'].append((d, stable_small[0][0][0], stable_small[0][1]) if stable_small else (d, np.nan, np.nan))
        branches['saddle'].append((d, unstable_small[0][0][0], unstable_small[0][1]) if unstable_small else (d, np.nan, np.nan))
        branches['consensus'].append((d, cons[0][0][0], cons[0][1]) if cons else (d, np.nan, np.nan))
    for k in branches:
        branches[k] = np.array(branches[k])
    dc_bis = delta_c_bisection(m, qB, pd)
    # fold: start from the trapped fixed point at the smallest delta where it exists
    ok = ~np.isnan(branches['trapped'][:, 1])
    d0 = branches['trapped'][ok][0, 0]
    z0 = [w for w in fixed_points(d0, m, qB, pd) if w[0][0] < 0.9 and w[1] < 0][0][0]
    dc_fold, z_fold = fold(m, qB, pd, d0, z0)
    print(f'q_B={qB}, p_d={pd}, m={m}: delta_c from bisection = {dc_bis:.5f}, from the fold = {dc_fold:.5f}; x_A* at the fold = {z_fold[0]:.4f}')
    print(f'E_c = (1-p_d) m delta_c = {(1 - pd) * m * dc_fold:.4f}')
    # ---- (c): trapping window E_c(pi) < E < E_max(pi) at m = 1 (only pi and E matter)
    pis = np.array([0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]) if not QUICK else np.array([0.0, 0.005, 0.01, 0.02, 0.04])
    Ec, Emax, pimax = {}, {}, {}
    for qb in [0.7, 0.8, 0.9]:
        lo, hi = [], []
        for pi in pis:
            if pi == 0.0:
                lo.append(Q_A / qb - 1.0); hi.append(np.inf)   # linear-stability threshold of the all-B state; no upper end
                continue
            pdv = pi / (1 + pi)               # pi = p_d / ((1-p_d) m) with m = 1  ->  p_d = pi/(1+pi)
            dc = delta_c_bisection(1.0, qb, pdv, tol=1e-3)
            if np.isnan(dc):
                lo.append(np.nan); hi.append(np.nan); continue
            lo.append((1 - pdv) * dc)
            hi.append((1 - pdv) * delta_max_bisection(1.0, qb, pdv, dc * 1.02))
        Ec[qb], Emax[qb] = np.array(lo), np.array(hi)
        pimax[qb] = pi_max(qb)
        print(f'q_B={qb}: pi_max = {pimax[qb]:.4f};  E_c(pi) = ' + ', '.join(f'{p:.4f}:{v:.3f}' for p, v in zip(pis, lo)))
        print(f'         E_max(pi) = ' + ', '.join(f'{p:.4f}:{v:.1f}' for p, v in zip(pis, hi)))
    np.savez(os.path.join(out_dir, 'figS_meanfield_threshold_data.npz'), deltas=deltas, **{k: v for k, v in branches.items()},
             dc_bis=dc_bis, dc_fold=dc_fold, pis=pis, **{f'Ec_{qb}': v for qb, v in Ec.items()}, **{f'Emax_{qb}': v for qb, v in Emax.items()}, **{f'pimax_{qb}': v for qb, v in pimax.items()})

    # ---- figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    ax = axs[0]
    tr, sa, co = branches['trapped'], branches['saddle'], branches['consensus']
    ax.plot(co[:, 0], co[:, 1], '-', color='#1f77b4', lw=2, label='consensus on A (stable)')
    ax.plot(tr[:, 0], tr[:, 1], '-', color='#d62728', lw=2, label='trapped state (stable)')
    ax.plot(sa[:, 0], sa[:, 1], '--', color='#7f7f7f', lw=1.8, label='saddle (unstable)')
    ax.axvline(dc_fold, color='k', lw=0.8, ls=':')
    ax.text(dc_fold + 0.03, 0.5, r'$\delta_c$', fontsize=12)
    ax.set_xlabel(r'deliberation time $\delta$ (mean-field units)', fontsize=11)
    ax.set_ylabel(r'fixed point $x_A^*$', fontsize=11)
    ax.set_ylim(-0.02, 1.05); ax.set_xlim(deltas[0], deltas[-1])
    ax.legend(fontsize=8.5, loc='center right', frameon=False)
    ax.text(-0.14, 1.04, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold')
    ax = axs[1]
    ax.plot(tr[:, 0], tr[:, 2], '-', color='#d62728', lw=2)
    ax.axhline(0, color='k', lw=0.8); ax.axvline(dc_fold, color='k', lw=0.8, ls=':')
    ax.set_xlabel(r'deliberation time $\delta$ (mean-field units)', fontsize=11)
    ax.set_ylabel('leading eigenvalue at the trapped state', fontsize=11)
    ax.set_xlim(deltas[0], deltas[-1])
    ax.text(-0.14, 1.04, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')
    ax = axs[2]
    cols = {0.7: '#2ca02c', 0.8: '#ff7f0e', 0.9: '#9467bd'}
    for qb in [0.7, 0.8, 0.9]:
        ok = ~np.isnan(Ec[qb])
        ax.plot(pis[ok], Ec[qb][ok], '-', color=cols[qb], lw=2.0, label=rf'$q_B={qb}$')
        okh = ok & np.isfinite(Emax[qb])
        ax.plot(pis[okh], Emax[qb][okh], '--', color=cols[qb], lw=1.4)
        ax.plot(0, Q_A / qb - 1, 's', ms=7, mfc='white', mec=cols[qb], mew=1.6, zorder=5)
        ax.axvline(pimax[qb], color=cols[qb], lw=0.8, ls=':')
    ok8 = ~np.isnan(Ec[0.8]) & np.isfinite(Emax[0.8])
    ax.fill_between(pis[ok8], Ec[0.8][ok8], Emax[0.8][ok8], color=cols[0.8], alpha=0.08, lw=0)
    ax.text(0.009, 4.0, 'trapped', color=cols[0.8], fontsize=10, ha='center', va='center')
    ax.text(0.009, 0.24, 'consensus on A', color=cols[0.8], fontsize=9, ha='center', va='center')
    ax.text(0.009, 150, 'consensus on A', color=cols[0.8], fontsize=9, ha='center', va='center')
    for pdv, mk in [(0.01, '^'), (0.02, 'v')]:   # parameter points of the paper at m = 1
        pi = pdv / (1 - pdv)
        for qb in [0.7, 0.8, 0.9]:
            dc = delta_c_bisection(1.0, qb, pdv, tol=1e-3)
            ax.plot(pi, (1 - pdv) * dc, mk, ms=7, color=cols[qb], zorder=6)
    ax.axhline(2.0, color='k', lw=0.8, ls='-.')
    ax.text(0.0462, 2.3, r'$E=2$ (largest $\delta$ in Fig. 4a and in the simulations)', fontsize=7.5, ha='right', va='bottom')
    h1, l1 = ax.get_legend_handles_labels()
    extra = [Line2D([], [], color='k', lw=2.0), Line2D([], [], color='k', lw=1.4, ls='--'), Line2D([], [], color='k', lw=0.8, ls=':'),
             Line2D([], [], marker='s', mfc='white', mec='k', ls='none', ms=6), Line2D([], [], marker='^', color='k', ls='none', ms=6), Line2D([], [], marker='v', color='k', ls='none', ms=6)]
    extra_l = [r'$E_c$ (lower end of trapping window)', r'$E_{\max}$ (upper end)', r'$\pi_{\max}$ (no trapping beyond)', r'$q_A/q_B-1$ (linear stability, $\pi\to 0$)', r'$p_d=0.01$, $m=1$', r'$p_d=0.02$, $m=1$']
    ax.legend(h1 + extra, l1 + extra_l, fontsize=7.6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3)
    ax.set_yscale('log')
    ax.set_xlabel(r'discovery-to-contact ratio $\pi = p_d/s$', fontsize=11)
    ax.set_ylabel(r'expected contacts per deliberation $E$', fontsize=11)
    ax.set_xlim(-0.001, 0.047); ax.set_ylim(0.08, 1000)
    ax.text(-0.14, 1.04, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figS_meanfield_threshold.pdf'), bbox_inches='tight', pad_inches=0.03)
    print('saved', os.path.join(out_dir, 'figS_meanfield_threshold.pdf'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='?', default='all', choices=['all', 'table', 'figure'])
    ap.add_argument('--out-dir', default='figures')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.which in ('all', 'table'):
        table()
    if args.which in ('all', 'figure'):
        main(args.out_dir, args.quick)
