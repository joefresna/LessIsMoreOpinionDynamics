"""
Mean-field model of best-of-2 collective decision-making with deliberation (Eq. 2 of the paper).

State variables (fractions of the population, summing to one):
    x_A  committed to A (transmitting)      d_A  deliberating on A (silent)
    x_B  committed to B (transmitting)      d_B  deliberating on B (silent)

Time unit: each agent attempts one update per unit time, i.e. one mean-field time unit corresponds to N timesteps
of the agent-based model, and a deliberation of delta time units to N*delta timesteps.

At each update an agent
  - with probability p_d discovers A independently: it adopts A and commits to it immediately
    (x_B, d_B and d_A all flow into x_A at rate p_d);
  - with probability 1 - p_d has a social contact. The opinion it is exposed to is drawn from the committed
    population with the quality-weighted probabilities
        p_A = q_A x_A / (q_A x_A + q_B x_B),   p_B = q_B x_B / (q_A x_A + q_B x_B).
    Exposure to the opposite opinion moves a committed agent into deliberation on that opinion and redirects a
    deliberating agent (restarting its deliberation). The network enters only through the effective rate m <= 1 of
    independent social contacts (m = 1 in the annealed limit), so the social rate is s = (1 - p_d) m.
Deliberation is completed at rate lambda = 1/delta.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

Q_A = 1.0


def rhs(t, y, lam, s, q_b, p_d, q_a=Q_A):
    """Right-hand side of Eq. (2). y = (x_A, d_A, x_B, d_B); lam = 1/delta; s = (1 - p_d) m."""
    x_a, d_a, x_b, d_b = y
    norm = q_a * x_a + q_b * x_b
    if norm < 1e-15:
        p_a = p_b = 0.5
    else:
        p_a, p_b = q_a * x_a / norm, q_b * x_b / norm
    return [lam * d_a - s * p_b * x_a + p_d * (x_b + d_b + d_a),
            s * p_a * (x_b + d_b) - lam * d_a - s * p_b * d_a - p_d * d_a,
            lam * d_b - s * p_a * x_b - p_d * x_b,
            s * p_b * (x_a + d_a) - lam * d_b - s * p_a * d_b - p_d * d_b]


def trajectory(delta, m, q_b, p_d, t_max, n_points=600, y0=(0.0, 0.0, 1.0, 0.0)):
    """Integrate Eq. (2) from y0 (default: all agents committed to B) up to t_max."""
    s = (1 - p_d) * m
    t_eval = np.linspace(0, t_max, n_points)
    return solve_ivp(rhs, [0, t_max], list(y0), args=(1.0 / delta, s, q_b, p_d), t_eval=t_eval,
                     method='LSODA', rtol=1e-10, atol=1e-12)


def asymptotic(delta, m, q_b, p_d, tol=1e-10, y0=(0.0, 0.0, 1.0, 0.0)):
    """Asymptotic state reached from y0: integrate in chunks until max |dy/dt| < tol."""
    lam, s = 1.0 / delta, (1 - p_d) * m
    y = np.array(y0, dtype=float); t = 0.0
    for chunk in [200, 800, 4000, 20000, 1e5, 1e6]:
        sol = solve_ivp(rhs, [t, t + chunk], y, args=(lam, s, q_b, p_d), method='LSODA', rtol=1e-10, atol=1e-12)
        y = sol.y[:, -1]; t += chunk
        if np.max(np.abs(rhs(0, y, lam, s, q_b, p_d))) < tol:
            break
    return y


def reaches_consensus(delta, m, q_b, p_d):
    """True if the population starting from all-B reaches consensus on A (x_A* > 0.5; in practice > 0.999)."""
    return asymptotic(delta, m, q_b, p_d)[0] > 0.5


def delta_c_bisection(m, q_b, p_d, d_min=0.02, d_max=6.0, tol=1e-4):
    """Critical deliberation time: smallest delta at which the all-B population no longer reaches consensus on A.
    Bisection on delta to within tol (mean-field time units). Returns nan if consensus is reached up to d_max."""
    if reaches_consensus(d_max, m, q_b, p_d):
        return np.nan
    lo, hi = d_min, d_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if reaches_consensus(mid, m, q_b, p_d):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_contacts(q_b, p_d, m=1.0, tol=1e-4):
    """E_c = (1 - p_d) m delta_c: critical expected number of independent contacts per deliberation period."""
    dc = delta_c_bisection(m, q_b, p_d, tol=tol)
    return (1 - p_d) * m * dc


# ---------------------------------------------------------------------------------------------------------------
# Fixed points, stability and the fold (saddle-node) of the trapped state, in reduced variables z = (x_A, d_A, d_B)
# ---------------------------------------------------------------------------------------------------------------
def reduced_rhs(z, lam, s, q_b, p_d):
    x_a, d_a, d_b = z
    f = rhs(0, [x_a, d_a, 1.0 - x_a - d_a - d_b, d_b], lam, s, q_b, p_d)
    return np.array([f[0], f[1], f[3]])


def jacobian(z, lam, s, q_b, p_d, h=1e-7):
    J = np.zeros((3, 3))
    for k in range(3):
        dz = np.zeros(3); dz[k] = h
        J[:, k] = (reduced_rhs(z + dz, lam, s, q_b, p_d) - reduced_rhs(z - dz, lam, s, q_b, p_d)) / (2 * h)
    return J


def fixed_points(delta, m, q_b, p_d):
    """All fixed points in the simplex, as a list of (z, leading eigenvalue), sorted by x_A."""
    lam, s = 1.0 / delta, (1 - p_d) * m
    found = []
    guesses = [(xa, da, db) for xa in [1e-3, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95, 0.999]
               for da in [0.0, 0.01, 0.05] for db in [0.0, 0.01, 0.05]]
    for g in guesses:
        r = root(reduced_rhs, np.array(g), args=(lam, s, q_b, p_d), method='hybr', tol=1e-13)
        if not r.success:
            continue
        z = r.x
        if np.max(np.abs(reduced_rhs(z, lam, s, q_b, p_d))) > 1e-10:
            continue
        if np.any(z < -1e-9) or 1 - z.sum() < -1e-9:
            continue
        if any(np.linalg.norm(z - w[0]) < 1e-6 for w in found):
            continue
        found.append((z, np.linalg.eigvals(jacobian(z, lam, s, q_b, p_d)).real.max()))
    return sorted(found, key=lambda w: w[0][0])


def fold(m, q_b, p_d, delta_guess, z_guess):
    """Saddle-node of the trapped state: solve F(z; delta) = 0, J(z; delta) v = 0, v.v0 = 1 for (z, v, delta)."""
    s = (1 - p_d) * m
    ev, V = np.linalg.eig(jacobian(z_guess, 1.0 / delta_guess, s, q_b, p_d))
    v0 = V[:, np.argmax(ev.real)].real
    v0 /= np.linalg.norm(v0)

    def G(w):
        z, v, d = w[:3], w[3:6], w[6]
        lam = 1.0 / d
        return np.r_[reduced_rhs(z, lam, s, q_b, p_d), jacobian(z, lam, s, q_b, p_d) @ v, v @ v0 - 1.0]
    r = root(G, np.r_[z_guess, v0, delta_guess], method='hybr', tol=1e-12)
    if np.max(np.abs(G(r.x))) < 1e-8:
        return r.x[6], r.x[:3]
    return np.nan, None
