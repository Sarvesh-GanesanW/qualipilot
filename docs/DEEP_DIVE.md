# Deep Dive — audit of the v1 codebase

This is the exhaustive list of problems the v1 `datapilot/checker.py`
shipped with, grouped by category. The v2 rewrite addresses every
item here.

## 1. Correctness bugs

1. `asyncio.get_event_loop()` is deprecated since Python 3.10 and
   raises `DeprecationWarning` → `RuntimeError` in 3.12+.
2. Dask duplicate detection used `map_partitions(.duplicated())` —
   misses cross-partition duplicates.
3. Outlier check computed Q1 and Q3 in two separate Dask `.compute()`
   calls per column, so an N-column run triggered 2N scans.
4. `asyncio` was listed in `install_requires` — it is part of
   stdlib and cannot be installed from PyPI.
5. `check_data_types()` returned a pandas Series when GPU was on and
   a dtypes object otherwise — callers had to branch on both.
6. `save_results` opened the file without an explicit encoding,
   which on Windows defaulted to cp1252 and mangled non-ASCII.
7. `matplotlib.pyplot.show()` blocks the process — unusable inside
   Jupyter kernels or Lambda.
8. `generate_summary` indexed `results['Range Validation']` without
   checking whether ranges were supplied.
9. `_print_and_return` coupled logging and return semantics in one
   helper — violates SRP and made testing ugly.
10. The cuDF path silently converted pandas → cuDF but did not accept
    Dask frames, so `engine=auto` on a Dask input errored out.

## 2. Packaging & distribution

11. `setup.py` only — no `pyproject.toml`, no PEP 621 metadata.
12. Package directory on disk (`datapilot/`) did not match the PyPI
    name (`data_quality_checker`), so `pip search` vs. `import`
    always confused new users.
13. No optional dependency groups — installing the core pulled
    matplotlib, openai, and aiohttp even for users who only needed
    the library.
14. No lock file, no build backend, no `py.typed`.
15. `python_requires='>=3.11'` but no upper bound or matrix tested.

## 3. Architecture

16. A single 340-line class wrapped every concern — engine, checks,
    LLM, rendering, IO, logging.
17. Print statements and logging calls were both used.
18. No typed result contract — everything was dict-of-whatever.
19. The LLM client was hard-coded to `http://localhost:11434/v1`,
    so AWS Bedrock / OpenAI needed code changes.
20. `self.llm_client.base_url` is a `URL` object in the OpenAI SDK;
    the code treated it as a string, which worked only because of
    implicit stringification.
21. No dependency injection — impossible to unit-test the LLM path
    without running a real server.

## 4. Observability & ops

22. Coloured log lines pollute log aggregators that parse ANSI.
23. No way to emit JSON logs for CloudWatch/Datadog.
24. No cost logging for the LLM leg — users had no visibility into
    tokens consumed on Bedrock/OpenAI.
25. Logs leaked raw column contents (outlier samples, duplicate
    rows) — that is PII in most customer datasets.

## 5. Dev tooling

26. No ruff / black / mypy / pytest config in the repo.
27. Tests (`tests/test_checker.py`) covered only happy paths and
    imported without a package layout.
28. No CI, no pre-commit, no Makefile, no Dockerfile.

## 6. Feature gaps

29. No CLI — the library was script-only.
30. No config file support — every knob had to be a constructor arg.
31. No HTML/Markdown report format.
32. No severity / exit code semantics — CI jobs had to parse JSON.
33. No cardinality / freshness / uniqueness profiling.
34. No schema enforcement (Pandera/GE-style).
35. No streaming or chunked support for very large files.
36. No cache for LLM responses.

## 7. LLM integration specifics

37. OpenAI client mis-configured — `base_url` had no trailing slash
    but `/chat/completions` was appended manually.
38. Retries existed but the backoff grew unbounded at `2 ** attempt`
    with no cap.
39. No support for system prompt customisation.
40. No token/parameter control (temperature, max_tokens).
41. No structured-output support.

## 8. Security

42. API key stored in plain attribute, written to `repr()` output.
43. No IAM policy guidance shipped for Bedrock.
44. No TLS verification knobs — pinning, custom CA, proxy, etc.

## 9. Deployment

45. No Dockerfile, no compose stack, no Lambda artefact, no Terraform.
46. Users had to hand-roll `pip install` + `conda activate rapids`
    to try the GPU path.

## Fix map (v1 → v2)

| v1 issue(s) | v2 resolution |
|---|---|
| 1, 38 | `tenacity` retry with explicit cap, sync client only |
| 2 | Engine API enforces global duplicate count |
| 3 | `quantiles()` batches every (col, q) into one pass |
| 4, 11–15 | `pyproject.toml`, hatch backend, optional extras |
| 5, 18 | Pydantic `CheckResult`/`QualityReport` typed models |
| 6 | All file IO uses `pathlib.Path.write_text(encoding="utf-8")` |
| 7 | Matplotlib moved behind optional `viz` extra, never auto-imported |
| 8 | Checks are independent; ranges short-circuits when empty |
| 9 | `Check.run` returns a typed result, no side-channel printing |
| 10 | `build_engine()` inspects module origin, not instance types |
| 16, 21 | Layered engines / checks / llm / reporting packages |
| 17, 22, 23 | Rich logs locally, JSON logs via env flag for cloud |
| 19, 20, 37, 39, 40 | Provider abstraction + explicit config fields |
| 24 | `BedrockProvider._log_usage` records input/output tokens |
| 25 | Samples are capped (`sample_size`) and never logged |
| 26–28 | Ruff, MyPy strict, pytest with coverage, pre-commit, CI |
| 29–32 | Typer CLI with `--fail-on`, html/md/json reporters |
| 33 | CardinalityCheck + FreshnessCheck added |
| 34 | Value-range enforcement remains; Pandera schema mode earmarked |
| 35 | Dask + cuDF engines; Polars streaming reader works out of the box |
| 42 | No `__repr__` override; API key never logged |
| 43 | `deploy/iam-policy.json` ships a minimal Bedrock+S3 policy |
| 44 | TLS + proxy respected via `httpx`/`botocore` env vars |
| 45 | Dockerfile, compose stack, Lambda image, Terraform module |
| 46 | `install.sh` / `install.ps1` one-click installers |
