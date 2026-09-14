# -*- coding: utf-8 -*-
"""
Step 2 — taxonomy overview (figure 1).

Output:
    outputs/fig1_taxonomy_overview.png
        category composition plus the 28 largest subclasses

Note: this step only shows what the database contains. An earlier version also
drew a fingerprint-based subclass similarity graph and a "chemical space"
landscape here; both were withdrawn after the fingerprint saturation discovered
in Step 3 (see fig3). They were measuring the fingerprint, not the chemistry.
"""

import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110
matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["savefig.bbox"] = "tight"

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


def main():
    df = pd.read_parquet(os.path.join(OUT, "lipids_clean.parquet"))
    with open(os.path.join(OUT, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    vc = df["CATEGORY"].value_counts()
    ys = np.arange(len(vc))[::-1]
    ax.barh(ys, vc.values, color=[cat_color(c) for c in vc.index], height=0.62)
    for y, v in zip(ys, vc.values):
        ax.text(v + 180, y, f"{v:,}", va="center", fontsize=10)
    ax.set_yticks(ys)
    ax.set_yticklabels([cat_short(c) for c in vc.index], fontsize=11)
    ax.set_xlim(0, vc.max() * 1.22)
    ax.set_xlabel("Number of entries")
    ax.set_title(f"Composition by category  ·  {len(df):,} entries", fontsize=12.5, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18, linestyle=":")

    ax2 = fig.add_subplot(gs[0, 1])
    sub = df.dropna(subset=["SUB_ABBREV"])
    cnt = sub.groupby("SUB_ABBREV").size().sort_values(ascending=False).head(28)
    catmap = (
        sub.groupby("SUB_ABBREV")["CATEGORY"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    ys = np.arange(len(cnt))[::-1]
    ax2.barh(ys, cnt.values, color=[cat_color(catmap.get(k)) for k in cnt.index], height=0.66)
    for y, v in zip(ys, cnt.values):
        ax2.text(v + 25, y, f"{v:,}", va="center", fontsize=8.5)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(cnt.index, fontsize=8.2)
    ax2.set_xlim(0, cnt.max() * 1.16)
    ax2.set_xlabel("Number of entries")
    ax2.set_title(
        f"28 largest subclasses  ·  {stats['n_subclass']} subclasses in total",
        fontsize=12.5, pad=12,
    )
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.18, linestyle=":")

    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_color(c), label=cat_short(c))
               for c in vc.index]
    ax2.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)
    fig.suptitle(
        "LIPID MAPS structure database: composition by category and subclass",
        fontsize=15, y=0.98,
    )
    p = os.path.join(OUT, "fig1_taxonomy_overview.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig1] {p}")


if __name__ == "__main__":
    main()
