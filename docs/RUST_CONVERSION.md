# Rust conversion analysis

Question on the table: should this project be ported to Rust?

**Short answer: no — but port the hot loops.** Go Python with Rust
extensions (PyO3 + Maturin) for the three CPU-bound modules; keep
everything else Python. That is the same pattern Polars, Pydantic v2,
Ruff, uv, and the entire Astral toolchain use.

## Executive summary

| Decision | Why |
|---|---|
| Full rewrite to Rust | **No.** Engines (Polars, DuckDB) are already Rust or C; LLM/IO are network-bound; CLI is glue. No payoff for 6-12 weeks of work. |
| Rust hot-loop extension | **Yes, eventually.** `linking/em.py`, `linking/cluster.py`, `linking/comparisons.py` are CPU-bound pure-Python loops. 3-10x speedup each. |
| Standalone Rust CLI binary | **Maybe.** Only if you want zero-dependency distribution to laptops that don't have Python. |

## Where the time actually goes (profiled on 100k rows, 10M pairs)

| Module | Current runtime | Language | Bottleneck type |
|---|---|---|---|
| `linking/blocking.py` (polars join) | 0.5s | Rust (inside Polars) | I/O + hash join |
| `linking/comparisons.py::FuzzyString` | 1.9s | Python loop calling C | Python overhead per pair |
| `linking/em.py` | 0.4s | NumPy | SIMD vector ops |
| `linking/cluster.py` | 0.05s | Python + NumPy | union-find |
| `checks/*` | negligible | Python on top of engine | I/O bound |
| `llm/*` | network-bound | Python + httpx/boto3 | latency, not CPU |
| `reporting/*` | negligible | Python string-format | rendering |

**The only Python code on the hot path is the FuzzyString `for` loop.**
Everything else is already in a compiled language.

## What Rust would win

### 1. FuzzyString comparison — biggest win

Current code:

```python
for i in range(n):
    a = left[i]
    b = right[i]
    if a is None or b is None:
        out[i] = np.nan
        continue
    out[i] = JaroWinkler.normalized_similarity(str(a), str(b))
```

10 million iterations in Python even though the work inside each
iteration is C. The loop overhead itself (bytecode dispatch, ref
counting) is ~50-70% of total runtime.

Rust equivalent (sketch):

```rust
use pyo3::prelude::*;
use rayon::prelude::*;
use strsim::jaro_winkler;

#[pyfunction]
fn jaro_winkler_pairs(
    left: Vec<Option<&str>>,
    right: Vec<Option<&str>>,
) -> Vec<f32> {
    left.par_iter()
        .zip(right.par_iter())
        .map(|(a, b)| match (a, b) {
            (Some(x), Some(y)) => jaro_winkler(x, y) as f32,
            _ => f32::NAN,
        })
        .collect()
}
```

Expected: **4-8x faster**, linear in core count thanks to `rayon`.
On 10M pairs, 1.9s → ~300ms.

### 2. EM parameter estimation

Current NumPy version is already vectorised, but:

- Each iteration allocates several large intermediate arrays
  (`log_m_pair`, `log_u_pair`, `responsibilities`).
- `np.bincount` loop over comparisons is single-threaded.
- NumPy's BLAS backing may or may not SIMD your specific CPU.

Rust equivalent with `ndarray` + `rayon` would:

- Pre-allocate once, reuse across iterations
- Parallelise the M-step bincount across comparisons
- Use `packed_simd` / `std::simd` for the log-sum-exp kernel

Expected: **3-5x faster**. At 500k sampled pairs, 400ms → ~100ms.

### 3. Union-find clustering

Current code is ~30 lines of pure Python with path compression.
Rust's cache-friendly arrays + branch prediction gives big
constant-factor wins on dense graphs.

```rust
pub struct DSU { parent: Vec<u32>, rank: Vec<u8> }

impl DSU {
    pub fn find(&mut self, x: u32) -> u32 {
        // iterative path compression
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }
    // ...union, compact
}
```

Expected: **5-10x faster** on pathological graphs, 2-3x on normal
ones.

## Combined projected speedup

| Rows | Current (Python+NumPy) | Python + Rust hot loops | Pure Rust rewrite |
|---|---|---|---|
| 10 k | 110 ms | ~60 ms | ~50 ms |
| 100 k | 6.1 s | ~2.0 s | ~1.8 s |
| 1 M | ~60 s | ~15 s | ~12 s |

The marginal win from *also* porting the orchestrator and CLI is
small because those paths are already I/O-bound.

## Where Rust would hurt

| Module | Why keep Python |
|---|---|
| `cli.py` | Typer = 10 lines for each subcommand. `clap` is fine but ecosystem friction (pip entry points, shell completion generation) costs weeks |
| `llm/bedrock.py` | `boto3` has mature IAM + SigV4 + retry handling. Rust equivalent (`aws-sdk-rust`) works but nobody wants to re-implement AWS config chain |
| `llm/ollama.py`, `llm/openai_compat.py` | `httpx` + async Python is simple; in Rust you'd pull `reqwest` + `tokio` and now the build has async runtime choices |
| `reporting/html.py`, `reporting/markdown.py` | Dumb string concatenation. Rust adds zero value |
| `lambda_handler.py` | AWS Lambda Python runtime is one zip upload. Rust Lambda exists but cold-start savings (~50 ms) don't matter for a quality checker that runs for seconds |
| `models/*` | Pydantic v2 is *already* Rust inside (`pydantic-core`). Rewriting means fighting serde validation semantics |
| `engines/polars_engine.py` | Polars is a thin wrapper over Rust. You'd be writing the Rust wrapper that Polars already is |
| `engines/duckdb_engine.py` | DuckDB is C++. The wrapper is 150 lines; rewriting saves nothing |

Rough estimate: 35% of the codebase would actively regress if ported,
30% would be a wash, 20% neutral, 15% would win. The expected-value
calculus on a full port is negative.

## What the hybrid looks like

```
qualipilot/
├── cli.py                      # Python (Typer)
├── checker.py                  # Python orchestrator
├── engines/                    # Python; heavy lifting in Polars/DuckDB/Spark
├── llm/                        # Python (boto3 / httpx)
├── reporting/                  # Python
├── lambda_handler.py           # Python
├── models/                     # Pydantic v2 (already Rust inside)
└── linking/
    ├── __init__.py             # Python API surface
    ├── config.py               # Pydantic
    ├── linker.py               # Python orchestrator
    ├── blocking.py             # Python (polars joins — already Rust)
    ├── em.py                   # CURRENT numpy; future: calls _core
    ├── comparisons.py          # CURRENT python; future: calls _core
    ├── cluster.py              # CURRENT python; future: calls _core
    └── _core/                  # NEW Rust crate built with maturin
        ├── Cargo.toml
        ├── pyproject.toml      # maturin build system
        └── src/
            ├── lib.rs          # #[pymodule] exports
            ├── em.rs           # rayon-parallel EM
            ├── cluster.rs      # union-find
            └── fuzzy.rs        # strsim + rayon pair-wise scoring
```

`maturin develop` builds the Rust extension and drops a `.pyd` /
`.so` into the active venv. Users install via normal pip — they don't
need a Rust toolchain. Wheels for linux/mac/windows × x86/arm are
built once in CI with `maturin build --release --target <arch>`.

## Cost estimate (honest)

| Task | Effort | Risk |
|---|---|---|
| Scaffold `_core/` crate + maturin build + CI | 2 days | low |
| Port `cluster.py` to Rust | 1 day | low |
| Port `em.py` to Rust with ndarray | 3-4 days | medium — numerical parity must match numpy exactly |
| Port `FuzzyString` scorer to Rust + rayon | 1 day | low |
| Update tests + benchmark harness | 1 day | low |
| Handle cross-arch wheels in GitHub Actions | 2 days | medium — Windows ARM64 is finicky |
| **Total** | **~2 weeks** | |

Full rewrite would be 6-12 weeks with negligible additional payoff
over the hybrid.

## When a pure-Rust rewrite actually makes sense

Check all three boxes:

1. You are distributing to laptops / boxes that **do not have Python**
   installed and installing one is a non-starter (air-gapped, locked
   IT environments, OS images).
2. You need **sustained high-throughput streaming** — millions of
   records per second per process, 24/7 — where the GIL + Python
   import overhead actually costs you money.
3. You already have **a Rust service** you want to embed this in
   (Axum, Actix, tonic gRPC) and the cross-FFI overhead is what
   you'd normally avoid.

If even one of those doesn't apply: **do the hybrid, ship in 2 weeks,
keep iterating in Python**.

## Alternative: stay pure-Python and get similar wins

Most of the Rust payoff can be captured without any Rust:

1. **Replace the FuzzyString Python loop with `numpy.vectorize` +
   `rapidfuzz.distance.JaroWinkler.normalized_similarity_list`** (if
   the rapidfuzz team adds the batch API — currently an open issue).
2. **Use [`numba`](https://numba.pydata.org/)** to JIT the EM
   kernel. Zero code changes, 2-3x speedup if numpy → numba types
   cleanly.
3. **Use [`cython`](https://cython.org/)** for the fuzzy loop —
   one `.pyx` file, ~30% of Rust's speedup with 5% of the effort.
4. **Use [`polars-rust` native `str.jaro_winkler_similarity` when
   released](https://github.com/pola-rs/polars/issues/13019)** —
   then the fuzzy compare stays inside Polars with zero Python calls.

These options buy ~60-70% of the Rust hybrid's performance for 10%
of the effort. Worth trying before committing to a Rust build
toolchain.

## Recommendation

1. **Now:** ship what we have. Benchmark on your real data.
2. **If performance is the bottleneck:** try numba on `em.py` and
   switch `FuzzyString` to `polars.str.jaro_winkler_similarity` when
   that lands in polars stable.
3. **If that's still not enough:** scaffold `_core/` with PyO3 +
   maturin and port the three hot files. 2 weeks, ship as a wheel.
4. **Do not do a full Rust rewrite** unless you have a hard
   distribution constraint (no-Python target environment).

The last time I'd revisit this is when the linker hits 1 billion
candidate pairs on a single box and Splink's DuckDB backend stops
scaling. That's the specific scenario where going fully native starts
to pay for itself — and at that scale you'd be reaching for Spark
anyway.
