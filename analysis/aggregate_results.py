"""
Aggregate one result set of bestOf2OnGraphs.py (one file per parameter combination, one convergence time per line,
equal to maxTime for runs that did not reach consensus) into a single CSV with one row per combination:
    N, mt, q, dp, net, p, dt, rw, seed, n_runs, n_success, median, times
where n_success counts runs with convergence time < mt and `times` lists all convergence times separated by ';'.
Files that differ only by a '_seed-<n>' suffix (the same combination split over several seeds) give separate rows.

Usage:  python aggregate_results.py <results_dir> <output.csv>
"""
import csv, glob, os, re, sys

PATTERN = re.compile(r"LIME_N-(?P<N>\d+)_mt-(?P<mt>\d+)_q-(?P<q>[0-9.]+)_dp-(?P<dp>[0-9.]+)_net-(?P<net>[a-z-]+)"
                     r"_p-(?P<p>[0-9.]+)_dt-(?P<dt>\d+)_rwp?-(?P<rw>[0-9.]+)(?:_seed-(?P<seed>\d+))?\.txt$")


def median(values):
    s = sorted(values); n = len(s)
    if n == 0:
        return float('nan')
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def aggregate(src, out):
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "LIME_*.txt"))):
        m = PATTERN.search(os.path.basename(f))
        if not m:
            print("skipping", f)
            continue
        times = [int(x) for x in open(f).read().split()]
        mt = int(m["mt"])
        rows.append([m["N"], mt, m["q"], m["dp"], m["net"], m["p"], m["dt"], m["rw"], m["seed"] or "", len(times),
                     sum(1 for t in times if t < mt), median(times), ";".join(map(str, times))])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "mt", "q", "dp", "net", "p", "dt", "rw", "seed", "n_runs", "n_success", "median", "times"])
        w.writerows(rows)
    print(out, len(rows), "rows")


if __name__ == "__main__":
    aggregate(sys.argv[1], sys.argv[2])
