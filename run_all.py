# -*- coding: utf-8 -*-
"""
Run the four steps in order.

Usage:
    python run_all.py

The input file must be in place first — see "Quick start" in the README.
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

STEPS = [
    ("01_build_data.py", "clean table, parse SMILES, Morgan fingerprints"),
    ("02_taxonomy.py", "category composition overview"),
    ("03_chain_graph.py", "headgroup / acyl-chain graphs"),
    ("04_null_models.py", "null models and permutation test"),
    ("05_stratified.py", "stratified permutation test"),
]


def main():
    for i, (script, desc) in enumerate(STEPS, 1):
        print(f"\n{'=' * 68}\n[{i}/{len(STEPS)}] {script} —— {desc}\n{'=' * 68}")
        r = subprocess.run([sys.executable, os.path.join(SRC, script)], cwd=SRC)
        if r.returncode != 0:
            print(f"\n!! {script} failed with exit code {r.returncode}")
            return r.returncode
    print("\nDone. Figures are written to outputs/; prebuilt copies live in figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
