# Put the input data here

This repository does not ship the input data. Download the **LIPID MAPS** structure
export (browse by category or use the bulk download), export it to xlsx, and place it
at `data/LipidMaps.xlsx`.

The scripts expect the worksheet that contains all lipid structure entries. In the
export used here that sheet is named `5大类脂质-get` — this string appears in
`src/config.py` and can be changed there or via `LIPIDMAPS_SHEET`.

## Required columns

Columns are matched **by position**, using `RAW_HEADER` in `src/01_build_data.py`.
Extra columns are ignored. The following are used:

Lipidmaps ID, INCHI_KEY, Name, Common Name, Sum Composition, Abbrev Chains,
ABBREVIATION, Chemical Formula, SMILES, Mono MS, M-H, M+H, M+NH4, M+Na, M+K,
Super Class / Class / Sub Class (HMDB), Category, Main Class, Sub Class,
Sub Class_Abbrev, HMDB ID, PubChem ID, CHEBI ID, KEGG ID, SwissLipids ID

If your export orders the columns differently, edit the `RAW_HEADER` constant. That is
the only place the column order is defined.

## Using a different location

```bash
set LIPIDMAPS_XLSX=D:\somewhere\LipidMaps.xlsx
set LIPIDMAPS_SHEET=5大类脂质-get
python run_all.py
```

On macOS or Linux use `export` instead of `set`. The defaults live in `src/config.py`.

## Licence note

LIPID MAPS data is governed by its own terms of use. This repository deliberately
contains **no data file and no derived data table** — only the processing scripts.
You run them locally and decide how to use the output. If you plan to redistribute a
derived table, check the current LIPID MAPS terms first.
