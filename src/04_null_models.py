# -*- coding: utf-8 -*-
"""
Step 4 — fixing the null model.

Step 3 (fig7) used the naive independence null: expected co-occurrence =
f(a) * f(b) / n. That null systematically under-estimates the expectation,
because the number of chains per molecule is fixed (2 or 3). Drawing chains into
a fixed number of slots is sampling WITHOUT replacement, not independent
sampling, so co-occurrence is sub-multiplicative by construction. Every PMI
comes out negative — an artefact, not evidence that chains repel each other.

This step does two things:
  fig8_null_models.png     the two null models side by side
  fig9_permutation_z.png   permutation test over the correct null

Output (written to outputs/ by default):
  fig8_null_models.png
  fig9_permutation_z.png
  null_models.json
  chain_pair_stats.csv
"""

import os
import json
import time
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
from config import OUT

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110
matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["savefig.bbox"] = "tight"

SEED = 7
N_PERM = 200
N_REPORT = 120


def load_pairs():
    m = import_module("03_chain_graph")
    df = m.prepare(pd.read_parquet(os.path.join(OUT, "lipids_clean.parquet")))
    d = df[df["HEAD"].notna() & (df["N_CHAIN"] >= 2)]
    chain_lists = []
    for cs in d["CHAINS"]:
        names = sorted({f"{c[0]}:{c[1]}" for c in cs})
        if len(names) >= 2:
            chain_lists.append(names)
    return chain_lists


def compute(chain_lists):
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

    keys = [k for k, v in pair.items() if v >= 40]
    kidx = {k: i for i, k in enumerate(keys)}
    obs = np.array([pair[k] for k in keys], dtype=float)

    # --- null model A: simple independence ---
    eA = np.array([freq[a] * freq[b] / n for a, b in keys], dtype=float)

    # --- null model B: chain count preserved (analytic approximation) ---
    ks = np.array([len(x) for x in chain_lists], dtype=float)
    K2 = float((ks * (ks - 1)).sum())
    eB = np.array([q[a] * q[b] * K2 for a, b in keys], dtype=float)

    # --- permutation test: simulate the null directly, no analytic shortcut ---
    rng = np.random.default_rng(SEED)
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
        if (rep + 1) % 50 == 0:
            print(f"  [perm] {rep + 1}/{N_PERM}")

    mu = null.mean(axis=0)
    sd = null.std(axis=0)
    z = (obs - mu) / np.maximum(sd, 1e-9)

    res = pd.DataFrame({
        "a": [chains[k[0]] for k in keys],
        "b": [chains[k[1]] for k in keys],
        "obs": obs, "expA": eA, "expB": eB, "null_mean": mu, "null_sd": sd, "z": z,
    })
    res["ratioA"] = res["obs"] / res["expA"]
    res["ratioB"] = res["obs"] / res["expB"]
    return res


def fig8(res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.24})

    ax = axes[0]
    ax.hist(res["ratioA"], bins=34, color="#C98B6B", edgecolor="white", linewidth=0.6)
    ax.axvline(1.0, color="#333333", lw=1.6, ls="--")
    ax.axvline(res["ratioA"].median(), color="#C0392B", lw=2.4)
    ax.text(1.03, ax.get_ylim()[1] * 0.90, "1.0 = independence", fontsize=9.5, color="#333333")
    ax.text(res["ratioA"].median() - 0.03, ax.get_ylim()[1] * 0.90,
            f"median {res['ratioA'].median():.2f}", fontsize=9.5,
            color="#C0392B", ha="right")
    ax.set_xlabel("Observed / expected")
    ax.set_ylabel("Number of chain pairs")
    ax.set_title("Null model A — simple independence\n"
                 "Every pair looks mutually repelled. It is an artefact.", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.hist(res["ratioB"], bins=34, color="#6B8EC4", edgecolor="white", linewidth=0.6)
    ax2.axvline(1.0, color="#333333", lw=1.6, ls="--")
    ax2.axvline(res["ratioB"].median(), color="#1F4E79", lw=2.4)
    ax2.text(1.03, ax2.get_ylim()[1] * 0.90, "1.0 = no preference", fontsize=9.5, color="#333333")
    ax2.text(res["ratioB"].median() + 0.03, ax2.get_ylim()[1] * 0.90,
             f"median {res['ratioB'].median():.2f}", fontsize=9.5, color="#1F4E79")
    ax2.set_xlabel("Observed / expected")
    ax2.set_ylabel("Number of chain pairs")
    ax2.set_title("Null model B — chain count preserved\n"
                  "The distribution returns to 1.0, but real dispersion remains", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Same data, different null model, opposite conclusion",
                 fontsize=14.5, y=1.02)
    p = os.path.join(OUT, "fig8_null_models.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig8] {p}")


def fig9(res):
    r = res.sort_values("z", ascending=False).reset_index(drop=True)
    top = r.head(N_REPORT)
    fig, ax = plt.subplots(figsize=(12.5, 9))
    x = np.arange(len(top))
    colors = ["#C0392B" if v > 0 else "#5B7FB5" for v in top["z"]]
    ax.bar(x, top["z"], color=colors, width=0.78)
    ax.axhline(0, color="#333333", lw=1.2)
    ax.axhline(3, color="#999999", ls=":", lw=1.2)
    ax.axhline(-3, color="#999999", ls=":", lw=1.2)
    ax.text(len(top) - 1, 3.4, "z = +3", fontsize=9, color="#888888", ha="right")
    ax.text(len(top) - 1, -3.9, "z = -3", fontsize=9, color="#888888", ha="right")

    for i in range(10):
        lab = f"{top.loc[i, 'a']} + {top.loc[i, 'b']}"
        ax.annotate(lab, (x[i], top.loc[i, "z"]), rotation=90, fontsize=8.4,
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    ax.set_xlabel("Chain pair (sorted by z-score)")
    ax.set_ylabel("Permutation z-score")
    ax.set_xlim(-1, len(top))
    ax.set_title(
        f"Permutation test over the corrected null ({N_PERM} reshuffles, chain count preserved)\n"
        f"{len(res)} pairs total, showing the {N_REPORT} largest z.  "
        f"z > 3: {int((res['z'] > 3).sum())} pairs   z < -3: {int((res['z'] < -3).sum())} pairs\n"
        f"The top pairs are canonical membrane-phospholipid chain combinations — but they\n"
        f"disappear once the data is stratified by headgroup (see README, trap B)",
        fontsize=12, pad=16,
    )
    ax.spines[["top", "right"]].set_visible(False)
    p = os.path.join(OUT, "fig9_permutation_z.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig9] {p}")


def main():
    t0 = time.time()
    cl = load_pairs()
    print(f"[load] {len(cl)} molecules with >= 2 chains")
    res = compute(cl)
    res.to_csv(os.path.join(OUT, "chain_pair_stats.csv"), index=False, encoding="utf-8-sig")
    fig8(res)
    fig9(res)

    top = res.sort_values("z", ascending=False).head(15)
    out = {
        "n_pairs": int(len(res)),
        "n_permutations": N_PERM,
        "ratioA_median": round(float(res["ratioA"].median()), 3),
        "ratioB_median": round(float(res["ratioB"].median()), 3),
        "z_gt_3": int((res["z"] > 3).sum()),
        "z_lt_minus3": int((res["z"] < -3).sum()),
        "top_enriched_pairs": [
            {"pair": f"{r.a} + {r.b}", "obs": int(r.obs),
             "null_mean": round(float(r.null_mean), 1), "z": round(float(r.z), 2)}
            for r in top.itertuples()
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "null_models.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] {out['runtime_sec']}s")


if __name__ == "__main__":
    main()
