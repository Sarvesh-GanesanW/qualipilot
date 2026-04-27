# Record linkage (probabilistic dedup)

qualipilot ships an in-house Fellegi-Sunter record linker. No
external dependency on splink — we implement the algorithm directly
with polars (blocking), rapidfuzz (string distance), and numpy (EM).

## When to use it

* deduplicating a customer table where typos and formatting
  variations hide real duplicates
* joining two systems that describe the same entities but share no
  primary key (bank accounts vs. CRM contacts, say)
* validating that an ETL join is not silently exploding rows

For exact row-level duplicates, `DuplicatesCheck` is faster — reach
for the linker only when exact comparison isn't enough.

## Core concepts

1. **Blocking rules** drop the N² problem to something tractable.
   Each rule is a list of columns whose values must all agree; two
   records "block together" if at least one rule matches. Pick rules
   where the largest bucket contains at most a few thousand rows.
2. **Comparisons** are per-column similarity buckets. Each returns a
   discrete **level** 0..k; level 0 is reserved for "null / no
   signal", higher levels mean stronger agreement.
3. **Fellegi-Sunter EM** learns, for every (comparison, level) pair,
   two probabilities: `m` (given a true match, how likely is this
   level?) and `u` (given a non-match, how likely is this level?).
   The ratio `log2(m / u)` gives that level's discriminative weight.
4. **Match probability** per pair combines the per-level weights and
   the learned prior λ. Pairs above a threshold form edges; a union-
   find pass yields entity clusters.

Everything runs in float32 with vectorised numpy, and the EM fit is
done on a random sample of the candidate pairs when the pair count
blows past 500k (`em_sample_size`). Scoring still uses every pair.

## Built-in comparisons

| kind | parameters | levels |
|---|---|---|
| `ExactMatch` | (none) | 3: null / different / exact |
| `FuzzyString` | `thresholds=(0.92, 0.80)` | null / low / mid / high |
| `NumericDiff` | `thresholds=(1.0, 5.0)` | null / far / within-5 / within-1 |

`FuzzyString` uses Jaro-Winkler from `rapidfuzz`. Pick thresholds
that reflect how aggressively you want to treat near-matches.

## Python API

```python
import polars as pl
from qualipilot.linking import (
    RecordLinker,
    LinkConfig,
    ExactMatch,
    FuzzyString,
    NumericDiff,
)

df = pl.read_csv("customers.csv")

cfg = LinkConfig(
    unique_id_column="customer_id",
    comparisons=[
        FuzzyString(column="name", thresholds=(0.92, 0.75)),
        ExactMatch(column="postcode"),
        NumericDiff(column="dob_year", thresholds=(0.0, 1.0)),
    ],
    blocking_rules=[
        ["postcode"],
        ["email"],
    ],
    match_threshold_probability=0.9,
)

result = RecordLinker(df, cfg).run()

print(result.summary())
print(result.match_pairs(0.9))        # DataFrame of high-confidence pairs
print(result.clusters[42])            # cluster id for record 42
print(result.parameters["m"])         # learned per-level m probs
print(result.timings_ms)              # stage-by-stage breakdown
```

## CLI

```bash
qualipilot link customers.csv \
    --id customer_id \
    --compare "name:fuzzy:0.92,0.75" \
    --compare "postcode:exact" \
    --compare "dob_year:numeric:0.0,1.0" \
    --block "postcode" \
    --block "email" \
    --threshold 0.9 \
    --output reports/customers.dedupe.json
```

The JSON output has three sections: `summary`, `matched_pairs` (each
row is a candidate with match probability above `--threshold`), and
`clusters` (map of record id to cluster id).

## Pipeline integration

Set `CheckConfig.linkage` inside your normal quality config and the
`LinkageCheck` fires alongside the rest:

```yaml
checks:
  missing_values: true
  duplicates: true
  linkage:
    unique_id_column: customer_id
    comparisons:
      - kind: fuzzy
        column: name
        thresholds: [0.92, 0.75]
      - kind: exact
        column: postcode
    blocking_rules:
      - [postcode]
    match_threshold_probability: 0.9
```

The result lands in `report.results[...]` with severity `warn` when
any probable-duplicate cluster is found.

## Performance notes

* **Blocking is the single biggest lever.** A rule on a column with
  10 distinct values over 100k records produces ~5 billion pairs.
  Tighter rules (postcode + surname prefix, say) drop that to
  ~100k in practice.
* **EM is fitted on a sample** when `candidate_pairs > em_sample_size`
  (default 500k). This keeps fitting time under a second even on
  tens of millions of pairs.
* **Comparisons are vectorised** where possible. String fuzzy scores
  use a tight rapidfuzz C loop; no Python-per-character work.
* Typical timings on a laptop (2026 CPU, no GPU):

  | Records | Candidate pairs | End-to-end |
  |---|---|---|
  | 10k | 100k | ~200 ms |
  | 100k | 2.5M (tight blocking) | ~2 s |
  | 1M | 5M (tight blocking) | ~10 s |

Tune `em_max_iter` downward if EM convergence is reliably fast on
your dataset; tune `em_tolerance` upward to stop sooner.
