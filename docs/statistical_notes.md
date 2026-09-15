# Statistical notes: why these tests, and what they protect against

This document records the reasoning behind the choices in `src/03_chain_graph.py`,
`src/04_null_models.py` and `src/05_stratified.py`. It is written for a reader who
works with machine learning but not necessarily with co-occurrence statistics.

## 1. Co-occurrence weight: why not MIC

Two candidates come up whenever the task is "find the relationships in this table":
MIC (maximal information coefficient) and PMI (pointwise mutual information). They
answer different questions, and only one of them fits an edge list.

| | PMI / O-E ratio | MIC |
|---|---|---|
| Unit of the score | one **pair of specific events** | one **pair of variables** |
| Output for a graph | one number per edge | one number per node pair, not per event pair |
| Designed for | discrete events, sparse co-occurrence | continuous variables, arbitrary relationship shape |
| Failure mode | biased upward for low-frequency pairs | needs enough samples to estimate a grid; not defined per event |

MIC was designed with *equitability* in mind: it aims to score a linear, periodic and
non-linear relationship comparably, and it does so by computing mutual information over
many grid resolutions and taking the maximum. That is the right tool when the question
is "are variable A and variable B related, however strangely". It is the wrong tool
when the question is "how strongly do the specific events 16:0 and 18:1 co-occur",
because the answer has to be attached to that specific edge, not to the chain-pair
variable as a whole.

In this dataset the variable is binary and sparse — "does chain X appear in this
molecule" — and the goal is a weighted edge per chain pair. PMI is pointwise by
construction, so it maps directly onto the edge:

```
PMI(a, b) = log[ P(a, b) / (P(a) * P(b)) ]
O/E(a, b) = observed(a, b) / expected(a, b) = exp(PMI(a, b))
```

`O/E` is used in the code because it reads more naturally than a logarithm, but it is
the same quantity.

**The known weakness of PMI.** PMI is biased upward for rare pairs: when `P(a)` and
`P(b)` are both tiny, the ratio is estimated from very few observations and inflates.
Two standard remedies exist — normalised PMI, and a significance test against an
explicit null. This repository uses both: NPMI where a bounded weight is needed, and a
permutation z-score for significance. A raw PMI ranking on sparse co-occurrence is not
trustworthy on its own.

## 2. The null model decides the answer

The first co-occurrence pass used independence as the null:

```
expected(a, b) = f(a) * f(b) / n
```

Under that null the median O/E is 0.702 and only 4.1% of pairs have a positive PMI:
every chain appears to repel every other chain. This is an artefact of the sampling
design, not a biological result. A lipid carries a **fixed** number of chains (2 or 3),
so chains are assigned to slots by sampling **without replacement**. Once 16:0 occupies
one slot, 18:1 has two remaining candidates instead of three, and the expected number of
co-occurrences is sub-multiplicative by construction.

The fix is a null that preserves the chain count per molecule. Under it the median O/E
returns to 1.087.

**The general rule**: before interpreting any "negative correlation" between items in
combinatorial data, check whether the number of items per sample is fixed. If it is,
independence is the wrong null and negative PMI carries no information about repulsion.

## 3. Simpson's paradox, and why it compounds with the sampling constraint

After correcting the null, some chain pairs survive with permutation z-scores around 20.
Stratifying by headgroup removes them:

| | pooled | stratified by headgroup |
|---|---|---|
| chain pairs tested | 440 | 511 |
| share with \|z\| > 3 | 34.3% | 0.2% |

Simpson's paradox is the name for the general pattern: an association that holds (or
appears to hold) in pooled data can vanish, or reverse, after conditioning on a
confounder — here, the headgroup.

The mechanism in this dataset is concrete. Different headgroups draw from different
chain pools: TG spans 12:0-22:1, while PC and PE sit on the 34-38 carbon membrane
lipids. Pooling them means testing two different distributions against one shared null,
which manufactures a batch of strong but meaningless co-occurrences.

Two things are worth separating, because they are often conflated:

1. **A confounder** — the headgroup shifts the chain distribution, so the pooled
   association is partly the headgroup difference in disguise.
2. **A structural constraint** — the fixed chain count makes co-occurrence
   sub-multiplicative regardless of biology.

Both are present here, and both had to be handled before any co-occurrence statistic
could be read as biology. Handling one and not the other still gives a wrong answer.

**The general rule**: whenever pooling is tempting, ask whether the pooled variable
hides sub-populations with different distributions. If so, stratify before interpreting
the pooled number — and report the stratified result even when it erases the effect, as
it does here.

## 4. What this means for a downstream graph model

These checks are not only about honesty in reporting. They change what a model would
learn.

A GNN or any structure-consuming model has no biological prior. It does not know that
TG and PC draw from different chain pools, and it has no way to know that "chains that
appear together" partly encodes the sampling rule rather than metabolism. If edges are
built without the checks above, the model will find the strongest regularities in the
graph and those regularities are:

- the fixed chain count (a sampling artefact), and
- the headgroup-specific chain preferences (a classification fact already known from
  biochemistry).

Training on such edges produces a model that re-derives lipid taxonomy at high accuracy
while learning nothing about the biology the analysis was meant to address. Removing
these components is therefore not a cosmetic correction; it is the step that makes the
remaining signal interpretable at all.

## 5. A note on the data source

LIPID MAPS is the reference structure database for lipids, so the low coverage of
relational fields is not a shortcoming of this particular export. It follows from what
the two kinds of resource are: a **structure** database enumerates and classifies
molecules, while **pathway** databases index prototypical members rather than species.
Expecting a structure database to supply pathway edges confuses the two roles, and no
alternative structure database would change that.
