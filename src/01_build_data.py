# -*- coding: utf-8 -*-
"""
Step 1 — turn the LIPID MAPS export into something a graph can be built from.

Input:
    the LIPID MAPS structure export (xlsx), worksheet name from config.SHEET

Output (written to outputs/ by default):
    lipids_clean.parquet   cleaned main table, ASCII column names
    fps_packed.npy         Morgan fingerprints, 2048 bit, np.packbits-packed
                           (only for rows whose SMILES parsed successfully)
    fps_valid.npy          boolean mask marking those rows
    stats.json             per-field coverage and category counts

This step only touches structure and taxonomy. No experimental data is involved.
"""

import os
import json
import time
import sys

import numpy as np
import pandas as pd
import openpyxl
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_XLSX as SRC, SHEET, OUT

OUT.mkdir(parents=True, exist_ok=True)

FP_BITS = 2048
FP_RADIUS = 2

# Columns of the upstream export, in positional order.
RAW_HEADER = [
    None, "Lipidmaps ID", "INCHI_KEY2", "Name", "Common Name",
    "Sum Composition_4for MSDIAL", "Common Name (No structure)_1", "Abbrev Chains_2",
    "Abbrev Chains_for MSDIAL_3", "Sum Composition", "Abbrev Chains", "ABBREVIATION",
    "Sum Composition_4", "Sum Composition_4for MSDIAL_2", "SYSTEMATIC_NAME", "SYNONYMS",
    "Chemical Formula", "INCHI_KEY", "INCHI", "SMILES", "Mono MS",
    "M-H", "M+H", "M+NH4", "M+Na", "M+K",
    "Super Class_HMDB", "Class_HMDB", "Sub Class_HMDB", "Category", "Main Class",
    "Sub Class", "Sub Class_Abbrev", "Lipidmaps ID_2", "HMDB ID", "PubChem ID",
    "CHEBI ID", "KEGG ID", "KEGG_ID (Class)", "SwissLipids ID", "LIPIDBANK_ID",
    "PLANTFA_ID", "CLASS_LEVEL4",
]

# Upstream column name -> short ASCII name used everywhere downstream.
KEEP = {
    "LM_ID": "Lipidmaps ID",
    "INCHIKEY": "INCHI_KEY",
    "NAME": "Name",
    "COMMON_NAME": "Common Name",
    "SUM_COMP": "Sum Composition",
    "ABBREV_CHAINS": "Abbrev Chains",
    "ABBREVIATION": "ABBREVIATION",
    "FORMULA": "Chemical Formula",
    "SMILES": "SMILES",
    "MONO_MASS": "Mono MS",
    "ADDUCT_M_H": "M-H",
    "ADDUCT_M_H_PLUS": "M+H",
    "ADDUCT_M_NH4": "M+NH4",
    "ADDUCT_M_NA": "M+Na",
    "ADDUCT_M_K": "M+K",
    "HMDB_SUPER": "Super Class_HMDB",
    "HMDB_CLASS": "Class_HMDB",
    "HMDB_SUB": "Sub Class_HMDB",
    "CATEGORY": "Category",
    "MAIN_CLASS": "Main Class",
    "SUB_CLASS": "Sub Class",
    "SUB_ABBREV": "Sub Class_Abbrev",
    "HMDB_ID": "HMDB ID",
    "PUBCHEM_ID": "PubChem ID",
    "CHEBI_ID": "CHEBI ID",
    "KEGG_ID": "KEGG ID",
    "SWISSLIPIDS_ID": "SwissLipids ID",
    "LIPIDBANK_ID": "LIPIDBANK_ID",
}

NA_TOKENS = {"", "na", "n/a", "none", "null", "nan", "-", "—"}


def is_na(v) -> bool:
    return v is None or str(v).strip().lower() in NA_TOKENS


def col_names_from_sheet(path, sheet):
    """Read the header row positionally so stable ASCII names can be assigned."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    header = next(ws.iter_rows(max_row=1, values_only=True))
    wb.close()
    names = []
    for i, h in enumerate(header):
        if h is None or str(h).strip() == "":
            names.append(None)
        else:
            names.append(str(h).strip())
        if i >= len(RAW_HEADER):
            break
    while len(names) < len(RAW_HEADER):
        names.append(None)
    print(f"[header] {len(header)} columns in file, using the first {len(RAW_HEADER)}")
    return names


def main():
    t0 = time.time()

    # ---------- 1. load ----------
    df = pd.read_excel(SRC, sheet_name=SHEET, dtype=str)
    df = df.iloc[:, : len(RAW_HEADER)]
    df.columns = [f"_col{i}" for i in range(len(RAW_HEADER))]
    print(f"[load] {df.shape[0]} rows x {df.shape[1]} columns")

    rename = {}
    for short, orig in KEEP.items():
        for i, h in enumerate(RAW_HEADER):
            if h == orig:
                rename[f"_col{i}"] = short
                break
    df = df[[c for c in df.columns if c in rename]].rename(columns=rename)
    print(f"[clean] kept {df.shape[1]} columns: {list(df.columns)}")

    for c in df.columns:
        df[c] = df[c].apply(lambda v: np.nan if is_na(v) else str(v).strip())

    before = len(df)
    df = df[~(df["LM_ID"].isna() & df["SMILES"].isna())].reset_index(drop=True)
    print(f"[clean] dropped {before - len(df)} empty rows -> {len(df)} rows")

    # ---------- 2. parse SMILES, compute fingerprints ----------
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_BITS
    )
    n = len(df)
    fps = np.zeros((n, FP_BITS // 8), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    n_bad = 0
    bad_examples = []

    for i, smi in enumerate(df["SMILES"].to_numpy()):
        if not isinstance(smi, str):
            n_bad += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_bad += 1
            if len(bad_examples) < 5:
                bad_examples.append(smi[:60])
            continue
        arr = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        fps[i] = np.packbits(arr)
        valid[i] = True

        if (i + 1) % 5000 == 0:
            print(f"[fp]   {i + 1}/{n}  parsed OK: {valid[:i + 1].sum()}")

    print(f"[fp] done: {valid.sum()}/{n} = {valid.sum() / n * 100:.1f}%  failed {n_bad}")
    if bad_examples:
        print(f"[fp] failure examples: {bad_examples}")

    # ---------- 3. write ----------
    df.to_parquet(os.path.join(OUT, "lipids_clean.parquet"), index=False)
    np.save(os.path.join(OUT, "fps_packed.npy"), fps)
    np.save(os.path.join(OUT, "fps_valid.npy"), valid)

    # ---------- 4. coverage stats ----------
    def cov(col):
        if col not in df.columns:
            return 0.0
        return round(float(df[col].notna().mean()), 4)

    stats = {
        "n_rows": int(n),
        "n_fp_valid": int(valid.sum()),
        "fp_valid_ratio": round(float(valid.sum() / n), 4),
        "fp_bits": FP_BITS,
        "fp_radius": FP_RADIUS,
        "coverage": {k: cov(k) for k in df.columns},
        "category_counts": {
            str(k): int(v) for k, v in df["CATEGORY"].value_counts(dropna=False).items()
        },
        "n_subclass": int(df["SUB_ABBREV"].nunique()),
        "n_main_class": int(df["MAIN_CLASS"].nunique()),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[coverage]")
    for k in ["SMILES", "INCHIKEY", "FORMULA", "ABBREV_CHAINS", "SUB_ABBREV",
              "KEGG_ID", "SWISSLIPIDS_ID", "HMDB_ID", "CHEBI_ID"]:
        print(f"   {k:16s} {stats['coverage'].get(k, 0) * 100:6.1f}%")
    print(f"\n[done] {stats['runtime_sec']}s  output dir {OUT}")


if __name__ == "__main__":
    main()
