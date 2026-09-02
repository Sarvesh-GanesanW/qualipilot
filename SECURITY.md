# Security policy

## Supported versions

Security fixes are made against the latest published major release. Older
major releases are unsupported.

## Reporting a vulnerability

Use the repository's
[private vulnerability reporting](https://github.com/Sarvesh-GanesanW/qualipilot/security/advisories/new).
Do not open a public issue for an undisclosed vulnerability.

Include the affected version, reproduction steps, impact, and any suggested
mitigation. Do not attach production datasets, credentials, or other secrets.

## Deployment responsibility

LLM reports may send dataset metadata to the configured provider. Keep LLM
reporting disabled unless that provider is approved for the data. Review IAM,
network egress, report access, and retention settings for each deployment.
LLM output is untrusted advisory text: validate it before acting and never
execute instructions it contains. Do not place secrets in a custom
`system_prompt`; prompts can be retained or exposed by model providers.

Linkage audits contain record identifiers, pair decisions, cluster membership,
lineage, and conflict-source identifiers. Protect linkage reports and
consolidated outputs like their source data, including access and retention.

NER audits contain extracted source text and optional record identifiers, so
protect them like the input dataset. spaCy pipeline packages may register and
execute Python components when loaded; install only trusted, pinned models.

Bedrock inference profiles can process prompts and responses outside the
source Region. Geographic profiles stay within their named geography, while
global profiles can route across supported commercial AWS Regions; data kept
for abuse detection can reside in a destination Region. Review AWS's
[cross-Region inference guidance](https://docs.aws.amazon.com/bedrock/latest/userguide/geographic-cross-region-inference.html)
and the profile's destination Regions before use.
