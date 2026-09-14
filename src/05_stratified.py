# -*- coding: utf-8 -*-
"""
Step 5 — stratified permutation test (trap B).

Step 4 found chain pairs with z-scores up to 20 after fixing the null model.
That looks like a biological signal. It is not.

The data is a mixture: different headgroups draw from completely different acyl
chain pools (TG spans 12:0-22:1, whereas PC/PE are concentrated on the 34-38
carbon membrane lipids). Pooling them means comparing two different
distributions against one shared null, which manufactures spurious strong
co-occurrences — a Simpson's-paradox situation.

Rerunning the same permutation test *within* each headgroup removes it.

Output (written to outputs/ by default):
  chain_pair_stats_stratified.csv   per-pair results, pooled across headgroups
  stratification.json               summary numbers
"""

import os
import json
import time
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
from config import OUT

SEED = 11
N_PERM = 100
MIN_MOL = 200      # a headgroup needs at least this many molecules
MIN_PAIR = 20      # a chain pair needs at least this many observations
MIN_PAIRS = 5      # and a headgroup needs at least this many usable pairs


def permutation_within(head, chain_lists, rng):
    chains = sorted({c for x in chain_lists for c in x})
    ci = {c: i for i, c in enumerate(chains)}
    N = len(chains)
    n = len(chain_lists)

    freq = np.zeros(N)
    for x in chain_lists:
        for c in x:
            freq[ci[c]] += 1
    S = sum(len(x) for x in chain_lists)
    q = freq / S

    mols = [([ci[c] for c in x], len(x)) for x in chain_lists]
    pair = {}
    for x in chain_lists:
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                k = (ci[x[i]], ci[x[j]])
                pair[k] = pair.get(k, 0) + 1

    keys = [k for k, v in pair.items() if v >= MIN_PAIR]
    if len(keys) < MIN_PAIRS:
        return None
    kidx = {k: i for i, k in enumerate(keys)}
    obs = np.array([pair[k] for k in keys], dtype=float)

    null = np.zeros((N_PERM, len(keys)))
    for rep in range(N_PERM):
        acc = np.zeros(len(keys))
        for ids, k in mols:
            pool = np.sort(rng.choice(N, size=k, replace=False, p=q))
            for i in range(k):
                for j in range(i + 1, k):
                    t = kidx.get((pool[i], pool[j]))
                    if t is not None:
                        acc[t] += 1
        null[rep] = acc

    mu = null.mean(axis=0)
    sd = null.std(axis=0)
    z = (obs - mu) / np.maximum(sd, 1e-9)
    return {
        "head": head,
        "n_molecules": n,
        "n_pairs": len(keys),
        "chains": chains,
        "keys": keys,
        "obs": obs,
        "null_mean": mu,
        "z": z,
    }


def main():
    t0 = time.time()
    m = import_module("03_chain_graph")
    df = m.prepare(pd.read_parquet(os.path.join(OUT, "lipids_clean.parquet")))
    d = df[df["HEAD"].notna() & (df["N_CHAIN"] >= 2)]
    rng = np.random.default_rng(SEED)

    rows = []
    per_head = []
    for head, g in d.groupby("HEAD"):
        cl = []
        for cs in g["CHAINS"]:
            names = sorted({f"{c[0]}:{c[1]}" for c in cs})
            if len(names) >= 2:
                cl.append(names)
        if len(cl) < MIN_MOL:
            continue
        r = permutation_within(head, cl, rng)
        if r is None:
            continue
        for i, k in enumerate(r["keys"]):
            rows.append((r["head"], r["chains"][k[0]], r["chains"][k[1]],
                         r["obs"][i], r["null_mean"][i], r["z"][i]))
        per_head.append({
            "head": head, "n_molecules": r["n_molecules"], "n_pairs": r["n_pairs"],
            "z_min": round(float(r["z"].min()), 2), "z_max": round(float(r["z"].max()), 2),
        })
        print(f"  {head:<8} molecules {r['n_molecules']:>5}  pairs {r['n_pairs']:>4}  "
              f"z range [{r['z'].min():.1f}, {r['z'].max():.1f}]")

    res = pd.DataFrame(rows, columns=["head", "a", "b", "obs", "null_mean", "z"])
    res.to_csv(os.path.join(OUT, "chain_pair_stats_stratified.csv"),
               index=False, encoding="utf-8-sig")

    out = {
        "n_headgroups_tested": len(per_head),
        "headgroups": per_head,
        "n_pairs": int(len(res)),
        "z_std": round(float(res["z"].std()), 3),
        "abs_z_gt_3": int((res["z"].abs() > 3).sum()),
        "abs_z_gt_3_pct": round(float((res["z"].abs() > 3).mean() * 100), 2),
        "unstratified_abs_z_gt_3_pct": 34.3,
        "note": ("Only headgroups with at least 200 molecules enter the test. "
                 "The unstratified comparison figure is taken from Step 4."),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "stratification.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] {out['runtime_sec']}s")


if __name__ == "__main__":
    main()
