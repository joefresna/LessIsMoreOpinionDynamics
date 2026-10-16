"""Helpers shared by the plotting scripts: loading aggregated result CSVs (see aggregate_results.py)."""
import numpy as np
import pandas as pd


def load_aggregated(path):
    """Read an aggregated CSV; adds the adaptation probability P = n_success / n_runs and the array of times."""
    df = pd.read_csv(path, dtype={'seed': str} if 'seed' in open(path).readline() else None)
    for col in ['q', 'dp', 'p', 'rw']:
        df[col] = df[col].astype(float).round(4)
    df['P'] = df.n_success / df.n_runs
    df['t'] = df.times.apply(lambda s: np.array([int(x) for x in str(s).split(';')]))
    return df


def select(df, **kw):
    """Rows matching all keyword conditions (floats compared with a tolerance)."""
    mask = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        if isinstance(v, float):
            mask &= np.isclose(df[k].astype(float), v)
        else:
            mask &= (df[k] == v)
    return df[mask]


def wilson(p, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h
