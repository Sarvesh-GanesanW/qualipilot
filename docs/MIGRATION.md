# Migration from v1.x

v2 is a major rewrite. The package name on PyPI is unchanged
(`qualipilot`), but the Python API, CLI, and configuration
are new. The old v1 code lives on the `v1` git tag if you need it.

## Quick translation

| v1 call | v2 replacement |
|---|---|
| `DataQualityChecker(df)` | `DataQualityChecker(df, QualipilotConfig())` |
| `checker.set_llm_api_key("ollama")` | `QualipilotConfig(llm=LLMConfig(provider="ollama", ...))` |
| `run_all_checks(column_ranges={...}, llm_model="qwen2:7b")` | `DataQualityChecker(df, QualipilotConfig(checks=CheckConfig(column_ranges={...}), llm=LLMConfig(provider="ollama", model="qwen2:7b"))).run()` |
| `checker.save_results(result, "x.json")` | `checker.save(report, "x.json")` — or set `QualipilotConfig.output_path` |
| `checker.visualize_outliers(...)` | removed; pull samples from `report.results[i].payload['per_column']` and plot with your tool of choice |

## Behavioural changes to expect

1. **Dask duplicates are now global**, not per-partition. Existing
   reports may surface more duplicates than before.
2. **Outlier quantiles use a single pass** per run; results match v1
   within float rounding.
3. **Logs are JSON** when `QUALIPILOT_JSON_LOGS=1` or `--json-logs`.
   Colours are opt-in via the Rich handler.
4. **LLM client is synchronous**. v1's async call path was never
   properly leveraged (we always awaited exactly one response);
   removing the async overhead simplifies error handling.
5. **Checks never print**. All output is on the returned
   `QualityReport`. Add a reporter (HTML/MD/JSON) or render it
   yourself.

## Upgrading a CI pipeline

Before:

```bash
python -m data_quality_checker_script data.csv  # custom wrapper
```

After:

```yaml
- name: data quality
  run: |
    qualipilot check data.csv \
        --config quality.yaml \
        --output reports/data.quality.html \
        --fail-on warn
```

The non-zero exit code on `warn` severity lets CI block merges
without you parsing JSON by hand.
