# Named-entity recognition

Qualipilot extracts labeled text spans through an explicitly installed spaCy
pipeline. This is named-entity recognition (NER): finding people,
organizations, locations, products, or domain-specific concepts in text. It
is separate from record linkage, which decides whether rows refer to the same
real-world entity.

## Install and choose a model

```bash
pip install "qualipilot[ner]"
python -m spacy download en_core_web_sm
```

The spaCy download command is convenient for local evaluation. For production,
pin the exact pipeline package in the application build, as recommended in
[spaCy's production model guidance](https://spacy.io/usage/models#production).
Qualipilot does not download a model or assume a language. `--model` must name
an installed package or a local pipeline directory containing an enabled
`ner` or `entity_ruler` component.

General-purpose pipelines reflect their training genre and label scheme. A
web-text English model is not evidence that clinical, legal, financial, or
multilingual text is handled correctly. spaCy likewise notes that statistical
entity predictions depend on their training examples and may need tuning.

## CLI

```bash
qualipilot ner notes.parquet \
  --text note \
  --id record_id \
  --model en_core_web_sm \
  --label PERSON \
  --label ORG \
  --output reports/notes.entities.json
```

CSV, Parquet, JSONL, and NDJSON inputs are supported. The text column must be
string-typed. Null text is skipped and counted. If `--id` is supplied, that
column must be unique and non-null. Otherwise `row_index` remains the stable
within-file reference. The command hashes the input before processing and
refuses to publish an audit if the file changes during extraction.

The JSON audit includes:

- input path and SHA-256;
- package and spaCy pipeline provenance;
- the normalized label filter, or `null` when all labels are retained;
- processed/null row and label counts;
- one item per entity with `row_index`, optional `record_id`, `text`, `label`,
  `start_char`, `end_char`, and optional `kb_id`.

Offsets are zero-based and half-open, so the original entity is
`document[start_char:end_char]`. They follow spaCy's documented `Span` API.
The audit contains extracted source text and may therefore contain personal or
sensitive data; protect it like the input dataset.

Pipeline packages can register and execute Python components when spaCy loads
them. Install models only from trusted sources, pin them in the application
build, and review them like any other executable dependency.

The CLI loads the input table and generated JSON audit in memory. `batch_size`
controls spaCy inference batching, not total report size; partition very large
corpora and merge their audits downstream.

## Python

```python
from qualipilot import SpacyEntityRecognizer

recognizer = SpacyEntityRecognizer(
    "en_core_web_sm",
    labels={"PERSON", "ORG"},
)
for entities in recognizer.extract_many(
    [
        "Ada Lovelace worked with Charles Babbage.",
        "OpenAI is in San Francisco.",
    ]
):
    for entity in entities:
        print(entity.to_dict())
```

`extract_many()` uses spaCy's ordered `Language.pipe()` batching path. Set
`batch_size` and `n_process` only after measuring representative documents.
Qualipilot does not invent confidence values because spaCy's ordinary
`Doc.ents` output does not provide calibrated per-entity probabilities.

## Release gate

Treat a model change as a data-contract change. Pin the model package, keep a
representative labeled holdout set, and compare exact-span, exact-label
precision, recall, and F-score before rollout. Check label-level results as
well as the aggregate: a good overall F-score can hide a weak business-critical
label. Also inspect tokenization coverage because unalignable reference spans
can be excluded from evaluation, as described in
[spaCy's training metrics documentation](https://spacy.io/usage/training#metrics).
