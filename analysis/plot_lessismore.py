"""
Fig. 2: the less-is-more effect in static networks (rho = 0). Adaptation probability (fraction of runs reaching
consensus on A within T) as a function of the Erdos-Renyi link probability p_k, for several deliberation times.
Dots: simulation results; lines: LOESS smooths (locally weighted quadratic regression, tricube weights, span 0.75);
bands: 95% confidence bands of the smooth.

Usage:  python plot_lessismore.py <aggregated_static.csv> [--N 50] [--q 0.7] [--pd 0.01] [--out fig2_lessismore.pdf]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from common import load_aggregated, select


def loess(x, y, x0s, span=0.75, degree=2):
    """Locally weighted polynomial regression and its 95% confidence band (as R's loess/predict.loess)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    k = int(np.ceil(span * n))

    def lrow(x0):  # row of the smoother matrix at x0: yhat(x0) = lrow(x0) @ y
        d = np.abs(x - x0); h = np.sort(d)[k - 1]
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        X = np.vander(x - x0, degree + 1, increasing=True)
        XtW = X.T * w
        return np.linalg.solve(XtW @ X, XtW)[0]

    L = np.array([lrow(xi) for xi in x])
    resid = y - L @ y
    IL = np.eye(n) - L
    M = IL @ IL.T
    delta1, delta2 = np.trace(M), np.trace(M @ M)
    sigma2 = resid @ resid / delta1
    dof = delta1 ** 2 / delta2
    R0 = np.array([lrow(x0) for x0 in x0s])
    yhat = R0 @ y
    se = np.sqrt(sigma2 * (R0 ** 2).sum(axis=1))
    tq = student_t.ppf(0.975, dof)
    return yhat, yhat - tq * se, yhat + tq * se


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--N', type=int, default=50)
    ap.add_argument('--q', type=float, default=0.7)
    ap.add_argument('--pd', type=float, default=0.01)
    ap.add_argument('--deltas', type=int, nargs='+', default=[0, 20, 40, 60, 80, 100])
    ap.add_argument('--out', default='fig2_lessismore.pdf')
    args = ap.parse_args()

    df = load_aggregated(args.csv)
    sub = select(df, N=args.N, q=args.q, dp=args.pd, rw=0.0)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    greys = plt.cm.Greys(np.linspace(0.35, 1.0, len(args.deltas)))
    for col, dt in zip(greys, args.deltas):
        s = sub[sub.dt == dt].sort_values('p')
        if s.empty:
            continue
        xs = np.linspace(s.p.min(), s.p.max(), 200)
        yhat, lo, hi = loess(s.p.values, s.P.values, xs)
        ax.fill_between(xs, np.clip(lo, 0, 1), np.clip(hi, 0, 1), color=col, alpha=0.15, lw=0)
        ax.plot(xs, np.clip(yhat, 0, 1), '-', color=col, lw=1.8, label=rf'$\delta={dt}$')
        ax.plot(s.p, s.P, 'o', ms=3.2, color=col, lw=0)
    ax.set_xlabel(r'link probability $p_k$', fontsize=12)
    ax.set_ylabel('adaptation probability', fontsize=12)
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(0.02, 0.32)
    ax.legend(title='deliberation time', frameon=False, fontsize=9, title_fontsize=10, loc='center right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches='tight', pad_inches=0.03)
    print('saved', args.out)
