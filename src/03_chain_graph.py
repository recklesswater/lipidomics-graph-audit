# -*- coding: utf-8 -*-
"""
Step 3 — build graphs from the domain-native representation.

Why not fingerprints: Step 2 originally built graphs from Morgan fingerprints.
That turned out to be unsound (see fig3) — a lipid is a small polar headgroup
attached to long alkyl chains, and a bounded-radius topological fingerprint
cannot see chain length. Radius 2 collapses 35,493 molecules into 9,070
distinct fingerprints.

So this step uses the domain-native representation instead:
    headgroup + fatty acyl chains + total carbons / total double bonds
All three come from the LipidMaps nomenclature itself. No experimental data.

Output (written to outputs/ by default):
  fig2_chain_cooccurrence.png       acyl-chain co-occurrence network
  fig3_fingerprint_failure.png      evidence that fingerprints saturate
  fig4_sum_composition_map.png      total carbons x total double bonds map
  fig5_headgroup_chain_network.png  headgroup-acyl chain bipartite graph
  fig6_tg_molecular_graph.png       molecule-level graph for the TG subclass
  fig7_cooccurrence_diagnostic.png  the naive null model (see Step 4)
  chains.json                       quantitative results
"""

import os
import re
import json
import time
import random
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import igraph as ig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110
matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["savefig.bbox"] = "tight"

SEED = 20260914
rng = np.random.default_rng(SEED)
ig.set_random_number_generator(random.Random(SEED))

CAT_COLORS = {
    "Glycerophospholipids [GP]": "#2E6F9E",
    "Fatty Acyls [FA]": "#D97706",
    "Glycerolipids [GL]": "#2F9E68",
    "Sphingolipids [SP]": "#B5457A",
    "Sterol Lipids [ST]": "#6B5BD2",
}
FALLBACK = "#8A8A8A"


def cat_color(c):
    return CAT_COLORS.get(str(c), FALLBACK)


def cat_short(c):
    s = str(c)
    return s.split("[")[-1].rstrip("]") if "[" in s else s[:6]


def igraph_layout(n_nodes, edges, niter=500):
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    lay = g.layout_fruchterman_reingold(niter=niter)
    xy = np.array(lay.coords, dtype=float)
    xy -= xy.min(axis=0)
    r = xy.max(axis=0)
    r[r == 0] = 1
    return xy / r


# ---------------------------------------------------------------- parsing
RE_ABBREV = re.compile(r"^\s*([^()]+?)\((.*)\)\s*$")
RE_CHAIN = re.compile(r"(\d+):(\d+)")
RE_SUMCOMP = re.compile(r"(\d+):(\d+)\s*$")


def parse_abbrev(s):
    """'TG(17:0_18:3_21:0)' -> ('TG', [(17,0,False),(18,3,False),(21,0,False)])"""
    if not isinstance(s, str):
        return None, []
    m = RE_ABBREV.match(s)
    if not m:
        return None, []
    head = m.group(1).strip()
    chains = []
    for part in re.split(r"[_/]", m.group(2)):
        cm = RE_CHAIN.search(part)
        if cm:
            ether = bool(re.match(r"^\s*[OoPp]-", part.strip()))
            chains.append((int(cm.group(1)), int(cm.group(2)), ether))
    return head, chains


def parse_sumcomp(s):
    """'TG 51:3' -> (51, 3)"""
    if not isinstance(s, str):
        return None
    m = RE_SUMCOMP.search(s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


# ---------------------------------------------------------------- prepare
def prepare(df):
    heads, chains = [], []
    for v in df["ABBREV_CHAINS"].to_numpy():
        h, c = parse_abbrev(v)
        heads.append(h)
        chains.append(c)
    df = df.copy()
    df["HEAD"] = heads
    df["CHAINS"] = chains
    df["N_CHAIN"] = [len(c) for c in chains]
    df["NC"] = [sum(c[0] for c in cc) if cc else np.nan for cc in chains]
    df["NDB"] = [sum(c[1] for c in cc) if cc else np.nan for cc in chains]

    sc = [parse_sumcomp(v) for v in df["SUM_COMP"].to_numpy()]
    df["SC_NC"] = [x[0] if x else np.nan for x in sc]
    df["SC_NDB"] = [x[1] if x else np.nan for x in sc]
    # prefer the parsed chains, fall back to the sum composition
    df["TOT_C"] = df["NC"].fillna(df["SC_NC"])
    df["TOT_DB"] = df["NDB"].fillna(df["SC_NDB"])
    df["MASS"] = pd.to_numeric(df["MONO_MASS"], errors="coerce")
    return df


# ---------------------------------------------------------------- fig3
def fig3_fingerprint_failure():
    """Show directly that different triglycerides share a radius-2 fingerprint."""
    radii = [1, 2, 3, 4, 5]
    uniq = [6953, 9070, 11661, 15140, 17647]
    total = 35493

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    ax.plot(radii, [u / total * 100 for u in uniq], "o-", color="#C0392B", lw=2.4, ms=8)
    for r, u in zip(radii, uniq):
        ax.annotate(f"{u:,}", (r, u / total * 100), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9.5)
    ax.axhline(100, ls="--", lw=1, color="#999999")
    ax.text(3.0, 101.5, "ideal: one fingerprint per molecule", fontsize=9, color="#777777")
    ax.set_xlabel("Morgan fingerprint radius")
    ax.set_ylabel("Unique fingerprints / total entries (%)")
    ax.set_ylim(0, 112)
    ax.set_xticks(radii)
    ax.set_title("Fingerprints saturate: a larger radius\nstill does not separate lipids", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, linestyle=":")

    ax2 = axes[1]
    ax2.set_axis_off()
    ax2.set_title("Example: three different triglycerides share\nan identical radius-2 fingerprint",
                  fontsize=12.5, pad=14)
    demo = [
        ("TG(12:0/12:0/18:2)", "C45H82O6", 718.61, 36),
        ("TG(12:0/12:0/20:2)", "C47H86O6", 746.64, 36),
        ("TG(12:0/12:0/20:3)", "C47H84O6", 744.63, 36),
    ]
    ax2.text(0.02, 0.86, "Molecule", fontsize=10.5, fontweight="bold")
    ax2.text(0.46, 0.86, "Formula", fontsize=10.5, fontweight="bold")
    ax2.text(0.68, 0.86, "Exact mass", fontsize=10.5, fontweight="bold")
    ax2.text(0.87, 0.86, "Bits set", fontsize=10.5, fontweight="bold")
    for k, (nm, fo, ms, nb) in enumerate(demo):
        y = 0.74 - k * 0.135
        ax2.text(0.02, y, nm, fontsize=10.5)
        ax2.text(0.46, y, fo, fontsize=10.5)
        ax2.text(0.68, y, f"{ms:.2f}", fontsize=10.5)
        ax2.text(0.89, y, str(nb), fontsize=10.5, color="#C0392B", fontweight="bold")
    ax2.plot([0.02, 0.98], [0.55, 0.55], color="#DDDDDD", lw=1)
    ax2.text(0.02, 0.40,
             "Why: the three molecules differ in the middle of an alkyl chain.\n"
             "At radius 2 only the two neighbouring bonds are visible, so every\n"
             "interior CH2 looks identical and chain length plus double-bond\n"
             "position are invisible to the fingerprint.\n\n"
             "Consequence: lipid graphs cannot be built from generic molecular\n"
             "fingerprints. Use the domain-native representation instead:\n"
             "headgroup + fatty acyl chains.",
             fontsize=10.5, va="top", linespacing=1.7)

    p = os.path.join(OUT, "fig3_fingerprint_failure.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig3] {p}")
    return {"radii": radii, "unique_fingerprints": uniq, "total": total}


# ---------------------------------------------------------------- fig2
def fig2_chain_cooccurrence(df, top_chain=42, min_pair=40, min_npmi=0.05):
    """
    Acyl-chain co-occurrence network, edge weight normalised with NPMI.

    Why not raw counts: the TG entries in LipidMaps are generated by enumerating
    chain combinations, so the most common pairs all end up with similar counts.
    That reflects the enumeration grid, not a biological preference. NPMI
    measures how far co-occurrence exceeds what independence would predict.
    (Step 4 shows that even this is not enough — see the README.)
    """
    d = df[df["HEAD"].notna() & (df["N_CHAIN"] >= 2)].copy()
    freq = {}
    pair = {}
    n_mol = 0
    for cs in d["CHAINS"]:
        names = sorted({f"{c[0]}:{c[1]}" for c in cs})
        if len(names) < 2:
            continue
        n_mol += 1
        for a in names:
            freq[a] = freq.get(a, 0) + 1
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                k = (names[i], names[j])
                pair[k] = pair.get(k, 0) + 1

    chains = [c for c, _ in sorted(freq.items(), key=lambda t: -t[1])[:top_chain]]
    cset = set(chains)
    idx = {c: i for i, c in enumerate(chains)}

    def npmi(a, b, v):
        pa, pb, pab = freq[a] / n_mol, freq[b] / n_mol, v / n_mol
        if pa <= 0 or pb <= 0 or pab <= 0 or pab >= 1:
            return 0.0
        return float(np.log(pab / (pa * pb)) / (-np.log(pab)))

    edges, ew = [], []
    for (a, b), v in pair.items():
        if a in cset and b in cset and v >= min_pair:
            s = npmi(a, b, v)
            if s < min_npmi:
                continue
            edges.append((idx[a], idx[b]))
            ew.append(s)

    xy = igraph_layout(len(chains), edges, niter=900)

    fig, ax = plt.subplots(figsize=(12.5, 10))
    ew_arr = np.array(ew)
    for (a, b), w in zip(edges, ew):
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                color="#B7BBC4", lw=0.35 + 3.2 * (w / ew_arr.max()) ** 0.6,
                zorder=1, alpha=0.65)
    vals = np.array([freq[c] for c in chains], dtype=float)
    dbs = np.array([int(c.split(":")[1]) for c in chains], dtype=float)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=60 + vals / vals.max() * 1500,
                    c=dbs, cmap="plasma", alpha=0.88,
                    edgecolors="white", linewidths=1.2, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Double bonds in the chain", fontsize=10)
    for c in chains:
        i = idx[c]
        ax.text(xy[i, 0], xy[i, 1], c, fontsize=8.0, ha="center", va="center",
                color="white", fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="#333333")])
    ax.set_axis_off()
    ax.set_title(
        f"Acyl-chain co-occurrence network  ·  {len(chains)} chains / {len(edges)} edges "
        f"(co-occurrence >= {min_pair})\n"
        f"Edge weight = NPMI, which discounts the fact that common chains pair up more often\n"
        f"Node size = chain frequency   ·   Colour = number of double bonds",
        fontsize=12, pad=16,
    )
    p = os.path.join(OUT, "fig2_chain_cooccurrence.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig2] {p}")

    scored = []
    for (a, b), v in pair.items():
        if a in cset and b in cset and v >= min_pair:
            scored.append((npmi(a, b, v), a, b, v))
    scored.sort(reverse=True)
    return {
        "n_chain_nodes": len(chains),
        "n_edges": len(edges),
        "n_molecules_with_chains": n_mol,
        "top_npmi_pairs": [
            {"a": a, "b": b, "npmi": round(s, 3), "count": int(v)}
            for s, a, b, v in scored[:12]
        ],
    }


# ---------------------------------------------------------------- fig4
def fig4_sum_map(df):
    d = df.dropna(subset=["TOT_C", "TOT_DB"]).copy()
    d = d[(d["TOT_C"] > 0) & (d["TOT_C"] < 100)]
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    for c in sorted(d["CATEGORY"].dropna().unique()):
        m = d["CATEGORY"] == c
        ax.scatter(d.loc[m, "TOT_C"], d.loc[m, "TOT_DB"], s=7, alpha=0.42,
                   color=cat_color(c), linewidths=0, label=cat_short(c))
    ax.set_xlabel("Total carbons (sum over acyl chains)", fontsize=11)
    ax.set_ylabel("Total double bonds (sum over acyl chains)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, frameon=False, markerscale=2.6)
    ax.grid(alpha=0.16, linestyle=":")
    ax.set_title(
        f"Sum-composition map  ·  {len(d):,} molecules\n"
        f"Both axes come from LIPID MAPS nomenclature; no experimental data involved",
        fontsize=13, pad=14,
    )
    ax.spines[["top", "right"]].set_visible(False)
    for x, y, t in [(34, 1, "PC/PE 34:1"), (38, 4, "PC 38:4"), (52, 3, "TG 52:3")]:
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(8, 8),
                    fontsize=9.5, color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8))
    p = os.path.join(OUT, "fig4_sum_composition_map.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig4] {p}")
    return {
        "n_mapped": int(len(d)),
        "carbon_range": [float(d["TOT_C"].min()), float(d["TOT_C"].max())],
        "db_range": [float(d["TOT_DB"].min()), float(d["TOT_DB"].max())],
    }


# ---------------------------------------------------------------- fig5
def fig5_headgroup_chain(df, top_chain=45):
    d = df[df["HEAD"].notna() & (df["N_CHAIN"] >= 1)].copy()
    rows = []
    for h, cs in zip(d["HEAD"], d["CHAINS"]):
        for c in cs:
            rows.append((h, f"{c[0]}:{c[1]}"))
    e = pd.DataFrame(rows, columns=["head", "chain"])
    w = e.groupby(["head", "chain"]).size().reset_index(name="n")
    w = w.sort_values("n", ascending=False).head(400)
    chains = [c for c in w.groupby("chain")["n"].sum().sort_values(ascending=False).index
              if c in set(w["chain"])][:top_chain]
    w = w[w["chain"].isin(chains)]
    heads = sorted(w["head"].unique())

    nH = len(heads)
    nodes = heads + chains
    hi = {h: i for i, h in enumerate(heads)}
    cidx = {c: nH + i for i, c in enumerate(chains)}
    edges = [(hi[h], cidx[c]) for h, c in zip(w["head"], w["chain"])]
    ew = w["n"].to_numpy(dtype=float)

    xy = igraph_layout(len(nodes), edges, niter=900)
    fig, ax = plt.subplots(figsize=(13, 10.5))
    for (a, b), ww in zip(edges, ew):
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                color="#C6C9D0", lw=0.3 + 2.6 * (ww / ew.max()), zorder=1, alpha=0.75)
    sizes = w.groupby("head")["n"].sum().reindex(heads).to_numpy(dtype=float)
    ax.scatter(xy[:nH, 0], xy[:nH, 1], s=sizes / sizes.max() * 1300 + 220,
               color="#C0392B", alpha=0.85, edgecolors="white", linewidths=1.3, zorder=3)
    sizes_c = w.groupby("chain")["n"].sum().reindex(chains).to_numpy(dtype=float)
    ax.scatter(xy[nH:, 0], xy[nH:, 1], s=sizes_c / sizes_c.max() * 420 + 45,
               color="#2E6F9E", alpha=0.80, edgecolors="white", linewidths=1.0, zorder=3)
    for h in heads:
        i = hi[h]
        ax.text(xy[i, 0], xy[i, 1], h, fontsize=8.2, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)
    for c in chains:
        i = cidx[c]
        ax.text(xy[i, 0], xy[i, 1], c, fontsize=6.4, ha="center", va="center",
                zorder=6, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#C0392B", markersize=11, label="headgroup"),
        plt.Line2D([], [], marker="o", ls="", color="#2E6F9E", markersize=8, label="acyl chain"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10.5, frameon=False)
    ax.set_axis_off()
    ax.set_title(
        f"Headgroup-acyl chain bipartite graph  ·  {nH} headgroups / {len(chains)} chains / "
        f"{len(edges)} edges\n"
        f"Edge width = number of molecules with that combination. "
        f"This is the skeleton that can be fed to a GNN.",
        fontsize=12.5, pad=16,
    )
    p = os.path.join(OUT, "fig5_headgroup_chain_network.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig5] {p}")
    return {"n_headgroups": nH, "n_chains": len(chains), "n_edges": len(edges)}


# ---------------------------------------------------------------- fig6
def fig6_tg_graph(df, head="TG", k=6):
    d = df[(df["HEAD"] == head) & (df["N_CHAIN"] >= 2)].reset_index(drop=True)
    if len(d) < 50:
        print(f"[fig6] skipped {head}")
        return {}
    d = d.sample(min(len(d), 1600), random_state=SEED).reset_index(drop=True)

    sets = [set(c[:2] for c in cs) for cs in d["CHAINS"]]
    S = np.zeros((len(sets), len(sets)), dtype=np.float32)
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a or not b:
                continue
            v = len(a & b) / len(a | b)
            S[i, j] = S[j, i] = v
    np.fill_diagonal(S, 0)
    idx = np.argsort(-S, axis=1)[:, :k]
    edges, seen = [], set()
    for i in range(len(sets)):
        for j in idx[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b and (a, b) not in seen and S[a, b] > 0:
                seen.add((a, b))
                edges.append((a, b))
    xy = igraph_layout(len(d), edges, niter=900)

    db = d["TOT_DB"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    for (a, b) in edges:
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                color="#D2D5DB", lw=0.28, zorder=1, alpha=0.5)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=16 + db * 3.2, c=db, cmap="plasma",
                    alpha=0.85, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Total double bonds", fontsize=10)
    ax.set_axis_off()
    ax.set_title(
        f"{head} molecular graph  ·  {len(d):,} molecules (random sample)\n"
        f"Edges = top-{k} Jaccard on acyl-chain composition   ·   "
        f"Node size and colour = total unsaturation",
        fontsize=12.5, pad=16,
    )
    p = os.path.join(OUT, f"fig6_{head.lower()}_molecular_graph.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig6] {p}")
    return {"head": head, "n_nodes": int(len(d)), "n_edges": len(edges)}


# ---------------------------------------------------------------- fig7
def fig7_cooccurrence_diagnostic(df, min_pair=40):
    """
    The naive independence null: expected co-occurrence = f(a) * f(b) / n.

    This is the WRONG null for this data and Step 4 corrects it. It is kept
    because the mistake is instructive: a fixed number of chain slots per
    molecule means sampling without replacement, which drives the ratio below 1
    everywhere and makes every chain pair look mutually repelled.
    """
    d = df[df["HEAD"].notna() & (df["N_CHAIN"] >= 2)]
    freq, pair, n = {}, {}, 0
    for cs in d["CHAINS"]:
        names = sorted({f"{c[0]}:{c[1]}" for c in cs})
        if len(names) < 2:
            continue
        n += 1
        for a in names:
            freq[a] = freq.get(a, 0) + 1
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                k = (names[i], names[j])
                pair[k] = pair.get(k, 0) + 1

    ratios, pmi = [], []
    for (a, b), v in pair.items():
        if v < min_pair:
            continue
        exp = freq[a] * freq[b] / n
        ratios.append(v / exp)
        pmi.append(np.log((v / n) / ((freq[a] / n) * (freq[b] / n))))
    ratios = np.array(ratios)
    pmi = np.array(pmi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"wspace": 0.26})
    ax = axes[0]
    ax.hist(ratios, bins=36, color="#6B8EC4", edgecolor="white", linewidth=0.6)
    ax.axvline(1.0, color="#C0392B", lw=2, ls="--")
    ax.text(1.02, ax.get_ylim()[1] * 0.92, "independence = 1.0",
            color="#C0392B", fontsize=10, ha="left")
    ax.set_xlabel("Observed co-occurrence / expected under independence")
    ax.set_ylabel("Number of chain pairs")
    ax.set_title("Chain co-occurrence sits almost entirely below 1\n"
                 "(only 3.4% of pairs fall in 0.8-1.25)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.hist(pmi, bins=36, color="#C98B6B", edgecolor="white", linewidth=0.6)
    ax2.axvline(0.0, color="#C0392B", lw=2, ls="--")
    ax2.text(0.03, ax2.get_ylim()[1] * 0.92, "PMI = 0 (no preference)",
             color="#C0392B", fontsize=10, ha="left")
    ax2.set_xlabel("PMI = ln[ p(a,b) / (p(a)p(b)) ]")
    ax2.set_ylabel("Number of chain pairs")
    ax2.set_title("PMI is negative for almost every pair, which looks\n"
                  "like mutual repulsion — it is not (see fig8)", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Diagnostic 1/2 — the naive independence null wrongly makes every chain pair look repelled",
        fontsize=13.5, y=1.02,
    )
    p = os.path.join(OUT, "fig7_cooccurrence_diagnostic.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig7] {p}")
    return {
        "n_pairs": int(len(ratios)),
        "ratio_median": round(float(np.median(ratios)), 3),
        "ratio_in_0.8_1.25_pct": round(float(((ratios > 0.8) & (ratios < 1.25)).mean() * 100), 1),
        "pmi_median": round(float(np.median(pmi)), 3),
        "pmi_positive_pct": round(float((pmi > 0).mean() * 100), 1),
    }


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    raw = pd.read_parquet(os.path.join(OUT, "lipids_clean.parquet"))
    df = prepare(raw)
    print(f"[load] {df.shape}")
    n_chain = int(df["HEAD"].notna().sum())
    print(f"[parse] headgroup + chains parsed for {n_chain}/{len(df)} "
          f"= {n_chain / len(df) * 100:.1f}%")
    print(f"[parse] total carbon available for {int(df['TOT_C'].notna().sum())} "
          f"= {df['TOT_C'].notna().mean() * 100:.1f}%")

    out = {}
    out["fig3_fingerprint"] = fig3_fingerprint_failure()
    out["fig2_chain_cooccurrence"] = fig2_chain_cooccurrence(df)
    out["fig4_sum_map"] = fig4_sum_map(df)
    out["fig5_bipartite"] = fig5_headgroup_chain(df)
    out["fig6_TG"] = fig6_tg_graph(df, "TG")
    out["fig7_cooccurrence_diagnostic"] = fig7_cooccurrence_diagnostic(df)
    out["coverage"] = {
        "n_total": int(len(df)),
        "n_with_head_and_chains": n_chain,
        "n_with_total_carbon": int(df["TOT_C"].notna().sum()),
    }
    out["runtime_sec"] = round(time.time() - t0, 1)

    with open(os.path.join(OUT, "chains.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[chains.json]")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] {out['runtime_sec']}s")


if __name__ == "__main__":
    main()
