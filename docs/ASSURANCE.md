# Assurance and standards mapping

Last reviewed: 2026-09-02.

Qualipilot implements technical controls informed by
[ISO/IEC 25012:2008](https://www.iso.org/standard/35736.html), the
[OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/),
[NIST AI 600-1](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence),
[SLSA v1.2](https://slsa.dev/spec/v1.2/), and the
[AWS Lambda best practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html).
This mapping is not certification, accreditation, or a claim that every
deployment conforms to those publications.

The statuses below mean:

- **supported**: a technical capability exists and is covered by automated
  checks;
- **partial**: useful controls exist, but domain or deployment evidence is
  still required;
- **external**: the deploying organization must supply the control; and
- **not applicable**: the cited risk is outside Qualipilot's architecture.

## Offered-feature assessment

| Feature | Baseline | Status | Implemented evidence | Required production evidence |
|---|---|---|---|---|
| Data-quality checks | ISO/IEC 25012 | Partial | Typed contracts, missing values, exact duplicates, dtypes, ranges, IQR outliers, cardinality, freshness, portable result schema | Business definitions, rule owners, thresholds, source truth, referential/cross-field rules, and acceptance criteria |
| Polars, Pandas, DuckDB, Dask, Spark | Portability and efficiency | Partial | Common engine contract, portable dtype families, parity tests, batched distributed aggregates, recorded benchmarks | Representative formats, schemas, distributions, cluster sizing, and service objectives |
| LLM narrative | OWASP LLM Top 10; NIST AI 600-1 | Partial | Default off, minimized bounded prompt, fixed untrusted-data instruction, escaped built-in sinks, no tools or agency, advisory label, provider/model provenance | Approved provider and data region, adversarial evaluation, factuality review, rate/spend limits, and incident ownership |
| Record linkage and consolidation | Statistical evaluation and traceability | Partial | Pair caps, deterministic sampling, fit diagnostics, unsafe-fit rejection, warning-fit consolidation gate, source/config/output hashes, lineage, precision/recall/F1 API | Representative labeled holdout, threshold calibration, subgroup analysis, drift monitoring, and human review of ambiguous pairs |
| Named-entity recognition | Model evaluation and provenance | Partial | Explicit pipeline loading, label validation, stable offsets, batching, text limits, spaCy/model metadata, input hash | Pinned model artifact hash, domain/language holdout, per-label and subgroup metrics, drift and model-change gates |
| Lambda operations | AWS Lambda best practices | Partial | Reused SDK client, immutable image digest, least-privilege IAM, encryption, version/ETag binding, idempotency, input limits, failure destination, metrics and alarms | Live IAM/encryption verification, representative load test, alarm exercise, replay test, backup restore, RTO and RPO |
| Release supply chain | SLSA v1.2 Build track | Build L2 provenance verified for v3.4.0 wheel and sdist | Locked dependencies, full-SHA Actions pins, CI gates, CodeQL, dependency/container scans, PyPI trusted publishing, and GitHub-hosted signed provenance tied to tag `v3.4.0` and commit `a1bc30a` | Consumers should verify downloaded artifacts; no Build L3 or SLSA Source-level claim is made |

## ISO/IEC 25012 data-quality characteristics

ISO/IEC 25012 defines a general model with 15 characteristics whose priority
depends on stakeholders and purpose. A library providing checks cannot by
itself establish that a dataset is fit for a particular business or legal use.

| Characteristic | Status | Qualipilot support and boundary |
|---|---|---|
| Accuracy | Partial | Dtype, range, and outlier rules detect configured anomalies; reference truth and semantic correctness are external |
| Completeness | Supported as a capability | Missing counts/percentages, minimum rows, and required columns; acceptable thresholds are domain-owned |
| Consistency | Partial | Dtype contracts and exact duplicates; referential and cross-field consistency are not built in |
| Credibility | External | Source authority, collection controls, attestations, and fitness claims belong to the data owner |
| Currentness | Supported as a capability | Timezone-aware freshness and future-timestamp detection; disabled until configured |
| Accessibility | Partial | Typed JSON plus HTML, Markdown, CLI, and Python APIs; no claim of formal accessibility conformance |
| Compliance | External/partial | Rules can encode requirements, but Qualipilot supplies no legislation- or sector-specific rule pack |
| Confidentiality | Partial | Samples and top values default off; LLM prompts omit paths, versions, errors, samples, and top values; metadata and reports can still be sensitive |
| Efficiency | Partial | Batched range, cardinality, null, and quantile operations plus distributed engines; workload sizing remains external |
| Precision | Partial | Exact versus approximate quantile provenance is reported; no universal measurement-uncertainty model |
| Traceability | Partial | Schema/package/time/source/config/provider/model provenance, durations, linkage lineage, and content hashes |
| Understandability | Partial | Structured results and human renderers; business glossary and rule rationale are external |
| Availability | Deployment-owned | Lambda concurrency, destinations, metrics, and alarms are supplied; availability objectives need live evidence |
| Portability | Partial | Five engines, portable dtype families, common formats, multi-OS and multi-Python CI |
| Recoverability | Deployment-owned | Atomic local writes, versioned S3 inputs, idempotent reports, failure queue, and digest rollback; restore objectives need exercises |

Safe public wording is: “Qualipilot provides controls that support selected
ISO/IEC 25012 data-quality characteristics.” Do not state that Qualipilot or a
dataset is ISO/IEC 25012 compliant without a scoped independent assessment.

## OWASP LLM Top 10 2025

| Risk | Status | Control and residual risk |
|---|---|---|
| LLM01 Prompt Injection | Partial | A fixed system instruction treats report content as untrusted. Dataset-derived names still reach the model, so real-provider adversarial testing remains necessary |
| LLM02 Sensitive Information Disclosure | Partial | Prompt minimization and default-off LLM use reduce exposure. Column names, dtypes, and aggregate findings still leave the process when enabled |
| LLM03 Supply Chain | Partial | Dependencies/actions are locked or pinned and models are explicit. Provider models and caller-installed spaCy pipelines remain supplier-controlled |
| LLM04 Data and Model Poisoning | Not applicable to library training | Qualipilot performs no training, fine-tuning, or retrieval augmentation; provider/model governance remains external |
| LLM05 Improper Output Handling | Supported in built-in sinks | HTML and Markdown escape untrusted content and CLI disables Rich markup. Raw API consumers must still treat `llm_report` as untrusted |
| LLM06 Excessive Agency | Architecturally mitigated | The LLM only returns narrative text and has no tools, write authority, or quality-gate control |
| LLM07 System Prompt Leakage | Partial | No built-in prompt secret is present and docs prohibit secrets in custom prompts; providers can still retain or disclose prompts |
| LLM08 Vector and Embedding Weaknesses | Not applicable | No vector store, embeddings, or RAG path exists |
| LLM09 Misinformation | Partial | Narratives are visibly advisory and record provider/model provenance; factuality and domain usefulness still require evaluation |
| LLM10 Unbounded Consumption | Supported as a capability | Prompt summaries, output tokens, timeouts, and retries are bounded; provider-side quotas and spend alerts are external |

## NIST AI RMF Generative AI profile

NIST AI 600-1 is voluntary organizational risk guidance, not a code
certification.

| Function | Status | Evidence and remaining ownership |
|---|---|---|
| GOVERN | Partial | Security policy, provider disclosure, release gates, and explicit limitations exist; organizations still need an AI-risk owner, inventory, risk register, review cadence, incident process, and approval records |
| MAP | Partial | Data sent to providers, intended uses, and feature limitations are documented; stakeholders, impacts, misuse cases, contracts, and regional requirements remain use-case specific |
| MEASURE | Partial | Deterministic tests, linkage metrics, and model-evaluation guidance exist; representative LLM, NER, linkage, subgroup, and adversarial acceptance results remain deployment-specific |
| MANAGE | Partial | LLM is opt-in, failures are structured, linkage consolidation fails closed on unsafe/warning fits by default, and Lambda has budgets/alarms; residual-risk acceptance and ongoing monitoring remain external |

## SLSA v1.2 release status

The Qualipilot v3.4.0 wheel and sdist were built once on GitHub-hosted Actions,
smoke-tested, attested, and published to PyPI through trusted publishing. Their
GitHub Sigstore attestations use the `https://slsa.dev/provenance/v1`
predicate and identify tag `v3.4.0`, commit
`a1bc30a303bf2cbefae3954aa70a54cbfd0db28d`, and
`.github/workflows/release.yml`.

Separate post-publication verification with GitHub CLI succeeded for both
published artifacts on 2026-09-02. Consumers can apply the same repository,
workflow, tag, commit, and hosted-runner constraints:

```bash
gh attestation verify qualipilot-3.4.0-py3-none-any.whl \
  --repo Sarvesh-GanesanW/qualipilot \
  --signer-workflow Sarvesh-GanesanW/qualipilot/.github/workflows/release.yml \
  --source-ref refs/tags/v3.4.0 \
  --source-digest a1bc30a303bf2cbefae3954aa70a54cbfd0db28d \
  --deny-self-hosted-runners

gh attestation verify qualipilot-3.4.0.tar.gz \
  --repo Sarvesh-GanesanW/qualipilot \
  --signer-workflow Sarvesh-GanesanW/qualipilot/.github/workflows/release.yml \
  --source-ref refs/tags/v3.4.0 \
  --source-digest a1bc30a303bf2cbefae3954aa70a54cbfd0db28d \
  --deny-self-hosted-runners
```

This Build L2 statement applies specifically to the verified v3.4.0 wheel and
sdist, not every historical or future artifact. The project makes no Build L3
claim and no SLSA Source-level claim because its source-control system does not
issue a Source VSA. PyPI trusted-publishing evidence is publish identity, not
build provenance.

## Statistical feature gates

Linkage probabilities and NER predictions are not universal accuracy claims.
For each production dataset:

1. version the input definition, linkage configuration or model package, and
   labeled holdout;
2. report precision, recall, F1, label/subgroup results, selected threshold,
   and known exclusions;
3. set acceptance gates before looking at a replacement model's results;
4. review linkage fit diagnostics and ambiguous pairs; and
5. rerun the gate on data drift, dependency changes, and every model or rule
   change.

`LinkageResult.evaluate_labeled_pairs()` counts blocked-out labeled matches as
false negatives. Consolidation rejects `rejected` and `warning` fits by
default; `allow_warning_fit=True` is an explicit, auditable override after
labeled evaluation. NER label filters are checked against labels actually
provided by the loaded pipeline.

## Operational responsibilities

Operational controls such as IAM, encryption keys, network egress, retention,
monitoring ownership, incident response, provider approval, human review,
recovery objectives, and legal obligations remain the deploying
organization's responsibility. Mocked tests cannot establish live AWS
permissions, delivery, alarms, availability, or restore behavior.

Linkage and NER audits may contain personal or otherwise sensitive source
content and identifiers. LLM prompts contain column names, dtypes, aggregate
findings, and check metadata. Classify and protect all three before enabling
retention or external processing.
