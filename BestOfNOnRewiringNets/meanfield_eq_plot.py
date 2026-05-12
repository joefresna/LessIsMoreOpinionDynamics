"""
fig_meanfield_all.py
─────────────────────
Generates all mean-field figures in one script, matching the style
and font sizes of the original script but with:
  - ODEs corrected to match Eq. (2) of the paper exactly
  - m(rho) = m_base*(1+alpha*rho) so rho=0 keeps full social interaction
  - Heatmap colormap matching the simulation Fig. 2 (viridis-like: dark purple → yellow)
  - Initial condition: 90% deliberating on B (paper text)
  - Parameters: qA=1.0, qB=0.8, D=0.01 (paper Fig. 1 caption)

Outputs:
  fig_mf_timecourse.pdf      — xA and xB time courses for varying delta
  fig_mf_heatmap.pdf         — heatmap of final xA in (rho, delta) space
  fig_mf_stackplot_A.pdf     — stacked area: rho=0.1, delta=2.0  (point A)
  fig_mf_stackplot_B.pdf     — stacked area: rho=0.8, delta=2.0  (point B)
  fig_mf_stackplot_C.pdf     — stacked area: rho=0.8, delta=0.1  (point C)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)

# ── Model configuration ────────────────────────────────────────────────────────
q_A    = 1.0
q_B    = 0.8
D      = 0.01          # paper Fig. 1 caption
m_base = 2.0
alpha  = 1.0           # m(rho) = m_base*(1 + alpha*rho)
T      = 50.0
t_eval = np.linspace(0, T, 500)

# Initial condition: 90% deliberating on B, 5% committed to each (paper text)
#y0 = [0.05, 0.0, 0.05, 0.90]
# all agents start committed to B
y0 = [0.0, 0.0, 1.0, 0.0]

# Sweep ranges (matching original script structure)
#delta_vals = np.array([0.005, 0.1, 0.2, 0.5, 2.0])
#delta_vals = np.array([0.005, 0.040, 0.075, 0.110, 0.150])
delta_vals = np.array([0.005, 0.050, 0.100, 0.150, 0.200])
delta_scale       = 500   # adjust this to match your simulation
delta_vals_display = (delta_vals * delta_scale).astype(int)
rho_vals   = np.linspace(0, 1, 20)

# ── Colours ────────────────────────────────────────────────────────────────────
# Time-course colours (Wong palette, same as original script spirit)
DELTA_COLORS = [
    '#E69F00',   # delta=0.005  amber
    '#56B4E9',   # delta=0.1    sky blue
    '#009E73',   # delta=0.2    teal green
    '#CC79A7',   # delta=0.5    pink
    '#D55E00',   # delta=2.0    vermillion
]

# Stacked area colours matching (xA blue, dA orange, xB green, dB red)
STACK_COLORS = ['#5B9BD5', '#ED7D31', '#70AD47', '#FF0000']
STACK_LABELS = ['$x_A$', '$d_A$', '$x_B$', '$d_B$']

# Heatmap colormap matching (dark purple → teal → yellow)
def _paper_cmap():
    colors = [
        (0.267, 0.005, 0.329),
        (0.282, 0.141, 0.458),
        (0.253, 0.265, 0.530),
        (0.163, 0.471, 0.558),
        (0.128, 0.566, 0.551),
        (0.157, 0.655, 0.522),
        (0.361, 0.737, 0.408),
        (0.678, 0.796, 0.196),
        (0.993, 0.906, 0.144),
    ]
    return LinearSegmentedColormap.from_list('paper_heat', colors, N=256)

PAPER_CMAP = _paper_cmap()

# Font sizes matching original script
FS_LABEL  = 24
FS_TICK   = 22
FS_LEGEND = 18
FS_CBAR   = 24
FS_CBAR_T = 28

# ── ODE system ─────────────────────────
def mean_field(t, y, D, lambda_, m):
    xA, dA, xB, dB = y
    norm = q_A * xA + q_B * xB
    if norm < 1e-15:
        pA, pB = 0.5, 0.5
    else:
        pA = q_A * xA / norm
        pB = q_B * xB / norm

    dxA = (lambda_ * dA
           - (1 - D) * m * pB * xA
           + D * (xB + dB))

    ddA = ((1 - D) * m * pA * (xB + dB)
           - lambda_ * dA
           - (1 - D) * m * pB * dA)

    dxB = (lambda_ * dB
           - (1 - D) * m * pA * xB
           - D * xB)

    ddB = ((1 - D) * m * pB * (xA + dA)
           - lambda_ * dB
           - (1 - D) * m * pA * dB
           - D * dB)

    return [dxA, ddA, dxB, ddB]

def m_of_rho(rho):
    return m_base * (1.0 + alpha * rho)

def simulate(D, delta, rho, y0):
    lambda_ = 1.0 / delta if delta > 0 else 1e6
    m       = m_of_rho(rho)
    sol = solve_ivp(
        mean_field, [0, T], y0, t_eval=t_eval,
        args=(D, lambda_, m),
        method='BDF', rtol=1e-8, atol=1e-10,
    )
    return sol.t, sol.y

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Time-course of adoption dynamics (xA and xB for varying delta)
# ════════════════════════════════════════════════════════════════════════════════
print('Generating fig_mf_timecourse.pdf ...')

plt.figure(figsize=(10, 6))

for i, delta in enumerate(delta_vals):
    t, sol = simulate(D, delta, rho=0.0, y0=y0)
    plt.plot(t, sol[0], linestyle='-',  color=DELTA_COLORS[i], linewidth=3.5)
    plt.plot(t, sol[2], linestyle='--', color=DELTA_COLORS[i], linewidth=3.5)

plt.xlabel('Time', fontsize=FS_LABEL)
plt.ylabel('Fraction Committed', fontsize=FS_LABEL)
plt.xticks(fontsize=FS_TICK)
plt.yticks(fontsize=FS_TICK)
plt.ylim(0, 1)

legend_handles = [
    Line2D([0],[0], color='black', linestyle='-',  linewidth=3, label='$x_A$'),
    Line2D([0],[0], color='black', linestyle='--', linewidth=3, label='$x_B$'),
]
for i, delta in enumerate(delta_vals):
    legend_handles.append(Patch(facecolor=DELTA_COLORS[i], label=f'$\\delta = {delta}$'))

plt.legend(handles=legend_handles, fontsize=FS_LEGEND, loc='best', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_mf_timecourse.pdf'), bbox_inches='tight')
plt.close()
print('  Done.')

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Heatmap of final xA in (rho, delta) space
# ════════════════════════════════════════════════════════════════════════════════
print('Generating fig_mf_heatmap.pdf ...')

heat_data = np.zeros((len(delta_vals), len(rho_vals)))
for i, delta in enumerate(delta_vals):
    for j, rho in enumerate(rho_vals):
        _, sol = simulate(D, delta, rho, y0)
        heat_data[i, j] = sol[0, -1]

# Flip for correct orientation (small delta at bottom)
heat_plot = heat_data[::-1]
y_labels  = [str(d) for d in delta_vals[::-1]]
x_labels  = [f'{r:.2f}' for r in rho_vals]

fig, ax = plt.subplots(figsize=(12, 8))

# Map delta values to uniform integer positions 0,1,2,3,4
# so imshow renders smoothly and ticks are equally spaced
n_delta = len(delta_vals)
n_rho   = len(rho_vals)

im = ax.imshow(
    heat_data,
    origin='lower',
    aspect='auto',
    cmap=PAPER_CMAP,
    vmin=0, vmax=1,
    interpolation='bilinear',
    extent=[-0.5, n_rho - 0.5, -0.5, n_delta - 0.5],
)

# Place ticks at uniform integer positions, label with actual delta values
ax.set_yticks(range(n_delta))
ax.set_yticklabels([str(d) for d in delta_vals_display], fontsize=FS_TICK)

# x ticks mapped from rho index to rho value
x_tick_rho = [0.0, 0.25, 0.5, 0.75, 1.0]
x_tick_idx = [(r - rho_vals.min()) / (rho_vals.max() - rho_vals.min()) * (n_rho - 1)
              for r in x_tick_rho]
ax.set_xticks(x_tick_idx)
ax.set_xticklabels([f'{r:.2f}' for r in x_tick_rho], fontsize=FS_TICK)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
cbar.ax.tick_params(labelsize=FS_CBAR)
cbar.set_label('Final $x_A$', fontsize=FS_CBAR_T)

ax.set_xlabel('Rewiring Probability ($\\rho$)', fontsize=FS_LABEL)
ax.set_ylabel('Deliberation Time ($\\delta$)', fontsize=FS_LABEL)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_mf_heatmap.pdf'), bbox_inches='tight')
plt.close()
print('  Done.')

# ════════════════════════════════════════════════════════════════════════════════
# FIGURES 3-5 — Stacked area plots for points A, B, C
# ════════════════════════════════════════════════════════════════════════════════
REGIMES = [
    dict(rho=0.1, delta=0.15, label='A'),
    dict(rho=0.9, delta=0.15, label='B'),
    dict(rho=0.9, delta=0.05, label='C'),
]

for regime in REGIMES:
    fname = f"fig_mf_stackplot_{regime['label']}.pdf"
    print(f'Generating {fname} ...')

    t, sol = simulate(D, regime['delta'], regime['rho'], y0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(t, sol, colors=STACK_COLORS,
                 labels=STACK_LABELS, alpha=0.92)

    ax.set_xlabel('Time', fontsize=FS_LABEL)
    ax.set_ylabel('Fraction of Population', fontsize=FS_LABEL)
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1])  # removes right white space
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=FS_LEGEND)
    fig.tight_layout(pad=0.5)  # minimal padding
    fig.savefig(os.path.join(OUT, fname),
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print('  Done.')

print('\nAll figures saved to:', OUT)
