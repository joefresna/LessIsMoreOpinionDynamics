"""
Average the per-run time series written by bestOf2OnGraphs.py with PLOT_EVO = True (evo/evo-LIME_..._<run>.txt,
columns t, x_A, d_A, x_B, d_B as numbers of agents) over runs. Runs that reached consensus before the horizon are
held in their final (absorbing) state. Output: one CSV per parameter combination with the fractions of the
population, subsampled every `step` timesteps.

Usage:  python aggregate_evo.py <evo_dir> <output_dir> <step>
"""
import glob, os, re, sys
import numpy as np


def aggregate(src, out, step):
    os.makedirs(out, exist_ok=True)
    pattern = re.compile(r"evo-(LIME_.*)_(\d+)\.txt$")
    groups = {}
    for f in glob.glob(os.path.join(src, "evo-LIME_*.txt")):
        m = pattern.search(os.path.basename(f))
        groups.setdefault(m.group(1), []).append(f)
    for base, files in sorted(groups.items()):
        mt = int(re.search(r"_mt-(\d+)", base).group(1))
        N = int(re.search(r"N-(\d+)", base).group(1))
        acc = np.zeros((mt, 4))
        for f in files:
            a = np.loadtxt(f, dtype=int)
            if a.ndim == 1:
                a = a[None, :]
            t, states = a[:, 0], a[:, 1:5].astype(float)
            full = np.full((mt, 4), np.nan)
            for ti, row in zip(t, states):
                if 1 <= ti <= mt:
                    full[ti - 1] = row
            last = states[0]
            for i in range(mt):          # forward-fill: hold the absorbing state after convergence
                if np.isnan(full[i, 0]):
                    full[i] = last
                else:
                    last = full[i]
            acc += full
        mean = acc / len(files) / N
        idx = np.arange(0, mt, step)
        np.savetxt(os.path.join(out, base + "_mean.csv"), np.column_stack([idx + 1, mean[idx]]), delimiter=",",
                   header="t,xA,dA,xB,dB", comments="", fmt=["%d", "%.5f", "%.5f", "%.5f", "%.5f"])
        print(base, "runs:", len(files), "final x_A:", round(mean[-1, 0], 3))


if __name__ == "__main__":
    aggregate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
