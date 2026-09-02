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
3. **Fellegi-Sunter EM** uses observed non-missing levels to learn two
   probabilities: `m` (given a true match, how likely is this level?) and
   `u` (given a non-match, how likely is this level?). Level 0 is excluded
   from parameter updates and contributes a neutral weight while scoring.
   For other levels, the ratio `log2(m / u)` gives that level's
   discriminative weight. A Jeffreys pseudocount keeps learned and scored
   probabilities away from unjustified exact zero and one.
4. **Match probability** per pair combines the per-level weights and
   the learned prior λ. Pairs above a threshold form edges; a union-
   find pass yields entity clusters.

EM fitting uses float32 NumPy arrays. When candidate count exceeds
`em_sample_size` (500,000 by default), fitting uses a deterministic
sample and scoring still evaluates every materialized candidate pair.

## Built-in comparisons

| kind | parameters | levels |
|---|---|---|
| `ExactMatch` | (none) | 3: null / different / exact |
| `FuzzyString` | `thresholds=(0.92, 0.80)` | null / low / mid / high |
| `NumericDiff` | `thresholds=(1.0, 5.0)` | null / far / within-5 / within-1 |

`FuzzyString` uses Jaro-Winkler from `rapidfuzz`. Pick thresholds
that reflect how aggressively you want to treat near-matches.
The DuckDB backend accepts ASCII fuzzy values only because its native
Jaro-Winkler function uses byte semantics for Unicode; use the default Polars
backend for international text.

## Python API

```python
import polars as pl
from qualipilot.linking import (
    ConsolidationConfig,
    RecordLinker,
    LinkConfig,
    ExactMatch,
    FuzzyString,
    MergeRule,
    NumericDiff,
    StringNormalization,
    SurvivorSortKey,
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
    normalization={
        "name": StringNormalization(),
        "postcode": StringNormalization(),
        "email": StringNormalization(),
    },
    match_threshold_probability=0.9,
)

deduplication = RecordLinker(df, cfg).deduplicate(
    ConsolidationConfig(
        sort_keys=(SurvivorSortKey(column="updated_at", descending=True),),
        completeness_columns=("email", "phone"),
        merge_rules={
            "email": MergeRule(strategy="first_non_null"),
            "phone": MergeRule(strategy="most_frequent"),
        },
    )
)
result = deduplication.linkage
clean = deduplication.consolidation

print(result.summary())
print(result.match_pairs(0.9))  # DataFrame of high-confidence pairs
print(result.clusters[42])  # cluster id for record 42
print(result.parameters["m"])  # learned per-level m probs
print(result.parameters["fit"])  # convergence and fit-safety diagnostics
print(result.timings_ms)  # stage-by-stage breakdown
clean.frame.write_parquet("customers.deduplicated.parquet")
print(clean.lineage)  # source id -> surviving id
```

The source frame remains unchanged. `StringNormalization` standardizes both
matching keys and consolidated values. It supports Unicode normalization,
trimming, whitespace collapsing, lowercasing, null tokens, and ordered regex
replacements for domain formats such as phone numbers. A normalization entry
may cover an output-only column as well as a comparison or blocking column.

Consolidation policy is explicit because survivor and merge choices are
business decisions. Survivor ranking applies sort keys in order, then
completeness, then the unique ID as a stable tie-breaker. Supported field
strategies are `survivor`, `first_non_null`, `most_frequent`, and `latest`
(with an `order_by` column). Calling `consolidate_records` directly preserves
the input schema. `RecordLinker.deduplicate` first normalizes configured
columns and preserves that normalized schema, so normalized string,
categorical, and enum columns have the `String` dtype. Both paths return one
row per cluster with source-to-survivor lineage and a metadata-only conflict
audit.

### Labeled-pair evaluation

[Splink's edge-evaluation guidance](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/edge_overview.html)
recommends ground-truth labels when measuring precision and recall or choosing
a match threshold. Evaluate a fixed threshold directly from a linkage result:

```python
labels = pl.DataFrame(
    {
        "__id_l__": [10, 10, 20],
        "__id_r__": [11, 12, 21],
        "is_match": [True, False, True],
    }
)
metrics = result.evaluate_labeled_pairs(labels, threshold=0.9)
print(metrics)
# {'threshold': 0.9, 'labeled_pairs': 3, 'tp': ..., 'fp': ...,
#  'tn': ..., 'fn': ..., 'precision': ..., 'recall': ..., 'f1': ...}
```

Pair IDs must use the same left/right orientation and dtypes as
`result.pairs`. Each pair must be unique, IDs and labels must be non-null, and
`is_match` must have Boolean dtype. A labeled pair absent from the candidate
set is treated as probability zero, so an omitted true match counts as a false
negative instead of disappearing from recall. Metrics with a zero denominator
are returned as `0.0`. These are edge metrics; evaluate clusters separately
when entity grouping is the production output.

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
    --survivor-sort "updated_at:desc" \
    --merge "phone:most_frequent" \
    --output reports/customers.dedupe.json \
    --deduplicated-output customers.deduplicated.parquet
```

The CLI normalizes every non-ID string field by default; pass `--raw-strings`
only when values are already canonical or case is meaningful.
When `--deduplicated-output` is present, the most complete record survives by
default and missing fields are filled from the highest-ranked record. Use
repeatable `--survivor-sort`, `--completeness`, and `--merge` options to
override those policies, or `--no-completeness` to disable populated-field
ranking. Sort keys use the dtype loaded from the input: CSV date strings sort
lexically, so use a parsed date/datetime column through the Python API or a
typed Parquet input when chronological order matters. The original input is
never overwritten.

The JSON output records schema/package versions, source and configuration
provenance, learned parameters, a summary, matched pairs, cluster membership,
consolidation rules, lineage, and a metadata-only merge audit. A JSON audit
path is required when writing a deduplicated dataset. The data file is
published first and the audit is written last as its commit marker. Consumers
must require the audit and verify `consolidation.output.sha256` before using
the data; this also detects a process or host failure between the two file
replacements.

The audit contains record identifiers, matched pairs, cluster membership,
lineage, and conflict-source identifiers. Treat it and the consolidated data
as sensitive as the input, with explicit access and retention controls.

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
any probable-duplicate cluster is found. A rejected linkage fit has severity
`error`.

## Operational notes

- With no blocking rules, linkage is a full cartesian comparison. The linker
  estimates that pair count before materializing candidates and stops when it
  exceeds `max_pairs_warning` (5 million by default). Supplying a higher value
  in `LinkConfig` is the explicit opt-in for a larger unblocked run.
- Blocking determines both recall and resource use. Low-cardinality rules
  can create a near-cartesian candidate set. Inspect candidate counts on a
  sample before increasing `max_pairs_hard_cap`.
- Blocking also changes the population used to fit comparison weights. Reusing
  a blocking field as a comparison can invert its learned agreement ordering;
  inspect fit diagnostics and labeled metrics rather than adding fields by
  intuition.
- The default hard cap is 50 million candidate pairs. That is a guardrail,
  not a memory guarantee; choose a lower cap for constrained workers.
- EM output and probability thresholds are not automatically calibrated to
  your domain. Evaluate labeled match and non-match pairs, then record the
  chosen configuration with downstream decisions.
- EM's two latent classes can swap labels or receive no identifying signal on
  small or homogeneous candidate sets. Qualipilot checks that higher declared
  agreement levels do not receive lower learned weights and that at least one
  comparison is informative. An unsafe fit is reported as `rejected`: its
  candidate probabilities are set to zero, its dedupe clusters remain
  singletons, and `RecordLinker.deduplicate()` refuses consolidation. A fit
  that reaches the iteration limit but passes those safety checks is scored
  with `fit.status == "warning"`. Consolidation refuses that fit by default;
  inspect `fit.warnings`, `converged`, and `max_parameter_delta`, then set
  `allow_warning_fit=True` (or CLI `--allow-warning-fit`) only when labeled
  evaluation supports the decision.
- Connected components are transitive: if A matches B and B matches C, all
  three records share a cluster even when A-C is below threshold.
- Use `scripts/bench_linking.py --n <rows>` on representative hardware and
  distributions. A benchmark failure exits nonzero; benchmark output is
  evidence for that run, not a general throughput guarantee.
- Preserve source records and review ambiguous pairs before publishing a
  consolidated dataset. Consolidation writes a separate frame or output file;
  it never mutates or deletes rows from the source.
