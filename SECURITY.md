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
