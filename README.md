# Lipidomics Graph Audit

**Three pre-flight checks before you build a graph over lipid structures.**

If you are about to point a GNN, a graph-learning model or a network analysis pipeline at a lipid structure database, run these three checks first. All three fail on the data used here, and each one hides a specific trap that is easy to walk into.

---

## Why this repository exists

Plenty of lipidomics papers mention graph neural networks. Very few say where the graph edges come from.

This repository takes one public lipid structure database and answers three questions honestly:

1. Do the relational fields actually carry enough information to build a graph?
2. Do standard molecular similarity measures still work on lipids?
3. Is co-occurrence computed from a structure database a real biological signal?

**All three come back negative.** But the debugging process is reusable: these three checks apply to any metabolite or small-molecule database, and each one has a characteristic failure mode.

Negative results are as useful as positive ones. They just need more care in the writing.

---

## Data boundary

This repository uses **only the public LIPID MAPS structure database**. There is no experimental data, no internal annotation, and no unpublished database anywhere in it.

| Item | Value |
|---|---|
| Input | LIPID MAPS Structure Database, exported to xlsx (not shipped here) |
| Entries | 35,501 -> 35,494 after cleaning |
| Categories covered | 5: GP, FA, GL, SP, ST |
| Not covered | PK (polyketides), SL (saccharolipids), PR (prenol lipids), among others |
| Coverage | **70.2% of the full LIPID MAPS database** (35,494 of 50,573 unique structures; LMSD total as of 2026-09) |

**Every number below holds for these 5 categories and 35,494 entries, not for all lipids.**

---

## Check 1 — do the relational fields carry anything?

| Field | Coverage |
|---|---|
| SMILES / InChIKey / chemical formula | ~100% |
| Sub Class_Abbrev | 81.7% |
| SwissLipids ID | 34.9% |
| CHEBI ID | 30.2% |
| HMDB ID | 19.2% |
| **KEGG ID** | **3.2% (1,126 / 35,494)** |

**This is a table that is rich in structure and almost empty in relations.**

The immediate consequence: if your plan was to use metabolic pathway membership as the GNN edge set, it cannot work here. 97% of the molecules have no pathway annotation at all.

**Why KEGG is that sparse, and the workaround.** KEGG does not index lipid species; it indexes *prototypical* lipids — one representative structure per lipid class or paradigm. A PC with 34 carbons and one with 38 carbons are the same KEGG entry, and adding or removing a double bond does not create a new one either. Species-level matching against KEGG therefore cannot succeed by construction, and a ~1% hit rate is the expected outcome, not a curation failure on our side.

The way around it is to bridge at the class level instead of the species level:

```
lipid species (no KEGG)
  -> subclass           (Sub Class_Abbrev, 81.7% coverage)
  -> a species in the same subclass that does have a KEGG ID
  -> its KEGG pathway
```

Every lipid in a subclass inherits the pathway membership of the subclass's prototypical member. Two caveats come with it: the inheritance is only as good as the assumption that all members of a subclass share pathway context, and the resulting edge carries **subclass-level resolution, not species-level**. For a heterogeneous graph that is usually an acceptable trade — the alternative is having no lipid-pathway edge at all — but the edge type should be labelled as class-inherited rather than direct, so that downstream analysis does not treat it as equivalent to an experimentally established link.

This bridge is also the cheapest source of the "lipid class ↔ pathway" edges that `figures/fig5_headgroup_chain_network.png` leaves missing.

**How to run this check**: print the coverage of every candidate relational field before you build the graph. Anything below roughly 50% cannot carry a primary edge type; it can only be a sparse feature.

---

## Check 2 — do molecular fingerprints separate lipids?

Taking Morgan fingerprints (2048 bit), how many distinct fingerprints do 35,493 molecules produce?

| Radius | Unique fingerprints | Share |
|---|---|---|
| 1 | 6,953 | 19.6% |
| 2 | 9,070 | 25.6% |
| 3 | 11,661 | 32.9% |
| 4 | 15,140 | 42.7% |
| 5 | 17,647 | 49.7% |

Even at radius 5, half the molecules still collide. **A concrete example** — all three of these have an identical radius-2 fingerprint with exactly 36 bits set:

| Molecule | Formula | Exact mass |
|---|---|---|
| TG(12:0/12:0/18:2) | C45H82O6 | 718.61 |
| TG(12:0/12:0/20:2) | C47H86O6 | 746.64 |
| TG(12:0/12:0/20:3) | C47H84O6 | 744.63 |

**Why**: a lipid is a small polar headgroup attached to long alkyl chains. A bounded-radius topological fingerprint only sees local environments, and every interior CH2 of a long chain looks identical within radius k. **Chain length and double-bond position are invisible to the fingerprint.**

**Consequence**: lipid graphs cannot be built from generic molecular fingerprints. Use the domain-native representation instead: headgroup, fatty acyl chains, total carbon count, total unsaturation. Incidentally, this also means any lipid similarity search built on molecular fingerprints deserves a re-run.

**How to run this check**: `np.unique(fingerprints, axis=0)` and count. If the unique count is far below the row count, the representation has already saturated on your data.

---

## Check 3 — is co-occurrence real?

Switching to the domain-native representation, the most natural edge rule is "two fatty acyl chains appear in the same molecule, so connect them". That fails too, and it fails twice, in two different ways.

### Trap A — the wrong null model turns "not enough slots" into "mutual repulsion"

The naive null model is independence: expected co-occurrence = f(a) * f(b) / n. Under it, the median observed/expected ratio is **0.702** and only 4.1% of pairs have a positive PMI. It looks as though every fatty acyl chain repels every other one.

That is an artefact. The number of chains per molecule is **fixed** (2 or 3), so chains are assigned by **sampling without replacement**, not independently: once 16:0 occupies a slot, 18:1 has 2 slots left instead of 3. Co-occurrence is sub-multiplicative by construction.

Switching to a null that preserves each molecule's chain count returns the median to **1.087** (`figures/fig8_null_models.png`).

> **A negative PMI is not a negative correlation, and it is not evidence of repulsion.** Here it only means the slots are finite.

### Trap B — pooling headgroups manufactures strong signals

After fixing the null, some chain pairs survive with permutation z-scores around 20 (`figures/fig9_permutation_z.png`). It looks like a real finding.

It is not. Rerunning the identical test **stratified by headgroup** removes it:

| | Pooled | Stratified by headgroup |
|---|---|---|
| Chain pairs | 440 | 511 |
| Pairs with \|z\| > 3 | **34.3%** | **0.2%** |
| Standard deviation of z | — | 0.392 |

The reason is that different headgroups draw from entirely different chain pools. TG spans 12:0-22:1; PC and PE sit on the 34-38 carbon membrane lipids. Pooling them means testing two different distributions against one shared null, which manufactures a batch of spurious strong co-occurrences: a textbook Simpson's paradox.

**Conclusion: acyl-chain co-occurrence in this database carries no usable biological signal.** A GNN trained on edges derived from a structure database's enumeration would learn the sampling constraints and the mixture of headgroup pools, and nothing about biology.

**How to run this check**: before any co-occurrence or correlation analysis, ask two questions. (1) Is the number of items per sample fixed? (2) Does the data mix subpopulations with different distributions? If both answers are yes, you need a different null and you need to stratify.

---

## What a graph model would have learned from this data

The three checks above are usually read as a caution about reporting. Their larger
consequence is about modelling.

A graph neural network carries no biological prior. It does not know that TG and PC draw
from different chain pools, and it cannot know that "chains that appear together" partly
encodes the sampling rule rather than metabolism. If edges are built from this database
without the checks above, the strongest regularities the model can find are:

* the fixed number of chains per molecule (a sampling artefact), and
* headgroup-specific chain preferences (a fact already established in biochemistry).

A model trained on those edges re-derives lipid taxonomy with high accuracy and learns
nothing about the biology the analysis was meant to address. Removing those components is
therefore not a cosmetic correction to a weight matrix; it is the step that makes any
remaining signal interpretable.

The statistical reasoning, including why co-occurrence is measured with pointwise mutual
information rather than MIC, why a fixed item count breaks the independence null, and how
the headgroup acts as a confounder, is written up in
[`docs/statistical_notes.md`](docs/statistical_notes.md).

On the data source: LIPID MAPS is the reference structure database for lipids, so the
sparse relational fields are not a shortcoming of this particular export. A structure
database enumerates molecules; a pathway database indexes prototypical members rather than
species. Expecting the first to supply the second conflates two kinds of resource, and no
alternative structure database would change that.

## What is actually reusable

It is not all bad news. These parts carry over directly:

| Output | Description |
|---|---|
| **Headgroup-acyl chain bipartite graph** | 23 headgroups / 45 chains / 376 edges, derived entirely from public structure, usable as a heterogeneous graph skeleton |
| **Domain-native features** | headgroup, acyl chains, total carbons, total unsaturation, exact mass. 42.2% of entries parse directly from nomenclature; the rest can be filled from sum composition |
| **The three diagnostic scripts** | the checks in `src/`, portable to another data source |
| **A reproducible negative result** | every number above is regenerated by `python run_all.py` |

Edges are still missing. There are only two legitimate sources left: **public pathway and ontology resources** (Reactome, WikiPathways, SwissLipids, the LIPID MAPS pathway layer) or **measured co-variation data**.

---

## Figures

| File | Content |
|---|---|
| `fig1_taxonomy_overview.png` | category composition and the 28 largest subclasses |
| `fig2_chain_cooccurrence.png` | acyl-chain co-occurrence network (only 15 edges survive NPMI) |
| `fig3_fingerprint_failure.png` | check 2: fingerprint saturation curve plus three colliding molecules |
| `fig4_sum_composition_map.png` | total carbons x total double bonds, 15,061 molecules |
| `fig5_headgroup_chain_network.png` | headgroup-acyl chain bipartite graph, the GNN-ready skeleton |
| `fig6_tg_molecular_graph.png` | molecule-level graph for the TG subclass, edges by acyl-chain Jaccard |
| `fig7_cooccurrence_diagnostic.png` | check 3, trap A: the naive null model and its misleading PMI |
| `fig8_null_models.png` | check 3, trap A corrected: the two null models side by side |
| `fig9_permutation_z.png` | check 3, trap B: the pooled permutation test |

---

## Quick start

```bash
pip install -r requirements.txt

# download the LIPID MAPS structure export to data/LipidMaps.xlsx
# see data/README.md for the required columns
python run_all.py
```

Each step also runs on its own from `src/`:

| Script | What it does | Approx. runtime |
|---|---|---|
| `01_build_data.py` | clean, parse SMILES, Morgan fingerprints | ~25 s |
| `02_taxonomy.py` | category composition overview | ~5 s |
| `03_chain_graph.py` | headgroup / acyl-chain graphs, figures 2-7 | ~7 s |
| `04_null_models.py` | null models and 200-shuffle permutation test | ~120 s |
| `05_stratified.py` | stratified permutation test (trap B) | ~35 s |

Input paths can be overridden with environment variables; see `src/config.py`.

---

## Layout

```
.
├── src/
│   ├── config.py            path configuration
│   ├── 01_build_data.py     cleaning, SMILES parsing, fingerprints
│   ├── 02_taxonomy.py       composition overview
│   ├── 03_chain_graph.py    headgroup / acyl-chain graphs
│   ├── 04_null_models.py    null models and permutation test
│   └── 05_stratified.py     stratified permutation test
├── figures/                 the nine figures
├── results/                 quantitative outputs (json / csv)
├── data/                    input data (not tracked, see data/README.md)
├── outputs/                 regenerated on each run (gitignored)
└── run_all.py               run everything
```

| File in `results/` | Content |
|---|---|
| `stats.json` | per-field coverage and category counts |
| `chains.json` | fingerprint saturation, co-occurrence network, bipartite graph |
| `null_models.json` | both null models and the pooled permutation test |
| `stratification.json` | the stratified permutation test |
| `chain_pair_stats.csv` | per chain pair: observed value, two expectations, z-score |
| `chain_pair_stats_stratified.csv` | the same, within each headgroup |

---

## Limitations

- Morgan fingerprints were computed without chirality, so stereochemistry (Z/E, sn-1/sn-2) does not contribute to the fingerprint.
- Headgroup and acyl-chain parsing depends on the `Abbrev Chains` field, covering 42.2% of entries and concentrated in glycerolipids and glycerophospholipids. Sphingolipids, sterols and fatty acyls are at 0%.
- The co-occurrence statistics treat all chains within a molecule as pairwise connected (a three-chain TG contributes three edges). That is precisely the source of trap A; it is reproduced and diagnosed here, not avoided.
- The stratified test only covers headgroups with at least 200 molecules (CL and TG, 511 chain pairs).
- The bipartite graph and the TG molecular graph are truncated for legibility (top 45 chains, 1,600 sampled molecules).
- Only 5 of the 8 LIPID MAPS categories are present, roughly half the database.

---

## Data and licence

- **Data**: LIPID MAPS Structure Database. This repository ships **no data files and no derived data tables**, only processing scripts. Using or redistributing LIPID MAPS data is subject to their terms.
- **Code**: MIT, see `LICENSE`.

---

---

## Citation

```
Chang, J. (2026). Lipidomics Graph Audit: three pre-flight checks before
building a graph over lipid structures.
https://github.com/recklesswater/lipidomics-graph-audit
```
