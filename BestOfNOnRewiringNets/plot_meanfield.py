"""
Figures of the mean-field model (Eq. 2):
  fig1_state_diagram.pdf  -- agent states and transitions (Fig. 1)
  fig4_meanfield.pdf      -- (a) asymptotic fraction committed to A over (m, delta), starting from all-B;
                             (b-d) time evolution in three representative regimes (Fig. 4)
Usage:  python plot_meanfield.py [diagram|meanfield|all] [--out-dir DIR]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from meanfield_model import asymptotic, delta_c_bisection, trajectory

Q_B, P_D = 0.8, 0.01     # parameters of Fig. 4
N_DISPLAY = 50           # population size used for the secondary axis in timesteps


# Colours of the three kinds of transition (Okabe-Ito palette, readable in greyscale and for colour-blind readers)
C_DISC, C_DELIB, C_SOC = '#009E73', '#D55E00', '#0072B2'


def state_diagram(fname):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_xlim(-1.0, 5.0); ax.set_ylim(-1.6, 2.8); ax.set_aspect('equal'); ax.axis('off')
    pos = {'xA': np.array([0.0, 2.0]), 'xB': np.array([4.0, 2.0]), 'dA': np.array([0.0, 0.0]), 'dB': np.array([4.0, 0.0])}
    lab = {'xA': r'$x_A$', 'xB': r'$x_B$', 'dA': r'$d_A$', 'dB': r'$d_B$'}
    R = 0.42
    for k, p in pos.items():
        ax.add_patch(Circle(p, R, facecolor='#7f7f7f', edgecolor='black', lw=1.5, zorder=3))
        ax.text(p[0], p[1], lab[k], ha='center', va='center', fontsize=24, zorder=4, color='white')

    def arrow(a, b, text, color, rad=0.0, side=0.0, t=0.5, off=0.0):
        """Arrow from circle a to circle b; `side` shifts it sideways, `rad` curves it. The label is placed on the
        arrow itself, at fraction t of its length, shifted by `off` perpendicular to it."""
        p1, p2 = pos[a].copy(), pos[b].copy()
        d = p2 - p1; u = d / np.hypot(*d); n = np.array([-u[1], u[0]])
        p1 = p1 + u * R + n * side; p2 = p2 - u * R + n * side
        ax.annotate('', xy=p2, xytext=p1, zorder=2,
                    arrowprops=dict(arrowstyle='-|>', lw=2.2, color=color, mutation_scale=20,
                                    connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))
        # point on the quadratic Bezier used by matplotlib's arc3 connection
        dx, dy = p2 - p1
        c = (p1 + p2) / 2 + rad * np.array([dy, -dx])
        q = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * c + t ** 2 * p2
        q = q + off * n
        ax.text(q[0], q[1], text, ha='center', va='center', fontsize=19, color=color, zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, lw=1.0))

    # independent discovery: from every other state into x_A
    arrow('xB', 'xA', r'$p_d$', C_DISC, side=0.0)
    arrow('dA', 'xA', r'$p_d$', C_DISC, side=-0.14, off=-0.38)
    arrow('dB', 'xA', r'$p_d$', C_DISC, rad=-0.12, t=0.28)
    # completion of deliberation
    arrow('dA', 'xA', r'$\lambda$', C_DELIB, side=0.14, off=0.36)
    arrow('dB', 'xB', r'$\lambda$', C_DELIB, off=-0.34)
    # social contact with a transmitting agent holding the opposite opinion
    arrow('xA', 'dB', r'$s\,p_B$', C_SOC, rad=-0.12, t=0.28)
    arrow('xB', 'dA', r'$s\,p_A$', C_SOC, t=0.28)
    arrow('dA', 'dB', r'$s\,p_B$', C_SOC, side=-0.21, t=0.6)
    arrow('dB', 'dA', r'$s\,p_A$', C_SOC, side=-0.21, t=0.6)
    ax.text(2.0, 2.62, 'committed (transmitting)', ha='center', va='center', fontsize=15, color='#444444')
    ax.text(2.0, -0.8, 'deliberating (silent)', ha='center', va='center', fontsize=15, color='#444444')
    handles = [plt.Line2D([], [], color=C_DISC, lw=2.2), plt.Line2D([], [], color=C_DELIB, lw=2.2),
               plt.Line2D([], [], color=C_SOC, lw=2.2)]
    ax.legend(handles, ['independent discovery', 'completion of deliberation', 'social contact'],
              loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=14, handlelength=1.4, columnspacing=1.2)
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def meanfield_figure(fname):
    m_vals = np.linspace(0.2, 1.0, 41)
    d_vals = np.linspace(0.02, 2.0, 60)
    heat = np.array([[asymptotic(d, m, Q_B, P_D)[0] for m in m_vals] for d in d_vals])
    dc = np.array([delta_c_bisection(m, Q_B, P_D, d_max=d_vals[-1]) for m in m_vals])
    Ec = (1 - P_D) * dc[-1]
    ok = ~np.isnan(dc)
    print(f'E_c (m = 1) = {Ec:.4f}; trapped state found for m >= {m_vals[ok].min():.3f} (delta <= {d_vals[-1]})')
    print('delta_c(m):', {round(float(m), 2): round(float(d), 3) for m, d in zip(m_vals[::5], dc[::5])})

    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.42, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(heat, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1,
                   extent=[m_vals[0] - 0.01, m_vals[-1] + 0.01, d_vals[0] - 0.0165, d_vals[-1] + 0.0165],
                   interpolation='nearest')
    ax.plot(m_vals, Ec / ((1 - P_D) * m_vals), color='white', lw=1.8, ls='--', label=r'$(1-p_d)\,m\,\delta = E_c$')
    ax.set_ylim(d_vals[0], d_vals[-1])
    ax.set_xlabel(r'effective rate of independent social contacts $m$', fontsize=15)
    ax.set_ylabel('deliberation time $\\delta$\n(mean-field units)', fontsize=15)
    ax.tick_params(labelsize=13)
    ax2 = ax.secondary_yaxis('right', functions=(lambda d: d * N_DISPLAY, lambda t: t / N_DISPLAY))
    ax2.set_ylabel(rf'$\delta$ (timesteps, $N={N_DISPLAY}$)', fontsize=15); ax2.tick_params(labelsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.1)
    cb.set_label(r'asymptotic $x_A$', fontsize=14); cb.ax.tick_params(labelsize=12)
    leg = ax.legend(loc='lower left', fontsize=12, framealpha=0.95, facecolor='#404040', edgecolor='none')
    for t in leg.get_texts():
        t.set_color('white')
    ax.text(0.42, 1.2, 'consensus on A', color='black', fontsize=14, ha='center', va='center')
    ax.text(0.84, 1.75, 'trapped in B', color='white', fontsize=14, ha='center', va='center')
    regimes = [(0.3, 1.5, 'b'), (1.0, 1.5, 'c'), (1.0, 0.04, 'd')]
    for m, d, l in regimes:
        mm = min(m, 0.985); dd = max(d, 0.06)
        ax.plot(mm, dd, 'o', ms=10, mfc='none', mec='red', mew=2.2, zorder=6)
        ax.text(mm - 0.035, dd, l, color='red', fontsize=18, fontweight='bold', ha='right', va='center', zorder=6)
    ax.text(-0.06, 1.04, 'a', transform=ax.transAxes, fontsize=20, fontweight='bold')

    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FF0000']
    labels = ['$x_A$', '$d_A$', '$x_B$', '$d_B$']
    for k, (m, d, l) in enumerate(regimes):
        axk = fig.add_subplot(gs[1, k])
        T = 150 if l != 'd' else 40
        sol = trajectory(d, m, Q_B, P_D, T)
        axk.stackplot(sol.t, sol.y, colors=colors, labels=labels, alpha=0.92)
        axk.set_xlim(0, T); axk.set_ylim(0, 1)
        axk.set_xlabel('time (mean-field units)', fontsize=13); axk.tick_params(labelsize=12)
        if k == 0:
            axk.set_ylabel('fraction of population', fontsize=13)
        axk.set_title(rf'$m={m}$, $\delta={d}$ ({int(round(d * N_DISPLAY))} timesteps)', fontsize=12)
        axk.text(-0.12, 1.08, l, transform=axk.transAxes, fontsize=20, fontweight='bold')
        if k == 2:
            axk.legend(loc='center right', fontsize=11, framealpha=0.9)
        print(f'regime {l}: final x_A = {sol.y[0, -1]:.4f}')
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='?', default='all', choices=['all', 'diagram', 'meanfield'])
    ap.add_argument('--out-dir', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.which in ('all', 'diagram'):
        state_diagram(os.path.join(args.out_dir, 'fig1_state_diagram.pdf'))
    if args.which in ('all', 'meanfield'):
        meanfield_figure(os.path.join(args.out_dir, 'fig4_meanfield.pdf'))
