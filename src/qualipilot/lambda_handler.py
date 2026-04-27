"""AWS Lambda entrypoint.

Expected event shape:

.. code-block:: json

    {
      "s3_uri": "s3://bucket/key.parquet",
      "config": { ...optional inline QualipilotConfig... }
    }

The handler downloads the object, runs the checker, and writes the
JSON report back to ``s3://<bucket>/<prefix>/report.json``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from qualipilot.checker import DataQualityChecker
from qualipilot.logging_setup import configure_logging
from qualipilot.models.config import QualipilotConfig

configure_logging(
    level=os.environ.get("QUALIPILOT_LOG_LEVEL", "INFO"),
    json_logs=True,
)
logger = logging.getLogger(__name__)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point. Returns a small JSON summary."""
    _ = context

    s3_uri = event.get("s3_uri")
    if not s3_uri:
        raise ValueError("event.s3_uri is required")

    cfg = QualipilotConfig(**event.get("config", {}))
    bucket, key = _parse_s3_uri(s3_uri)

    import boto3

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmpdir:
        local = Path(tmpdir) / Path(key).name
        logger.info("downloading s3://%s/%s", bucket, key)
        s3.download_file(bucket, key, str(local))

        checker = DataQualityChecker(local, cfg)
        report = checker.run()

    output_key = event.get("output_key") or _derive_output_key(key)
    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=report.to_json().encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("report written to s3://%s/%s", bucket, output_key)

    return {
        "bucket": bucket,
        "input_key": key,
        "output_key": output_key,
        "summary": {r.name: r.severity for r in report.results},
    }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"expected s3:// uri, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _derive_output_key(input_key: str) -> str:
    # keep the report next to the source, but in a reports/ prefix
    stem = Path(input_key).stem
    return f"reports/{stem}.quality.json"


# allow `python -m qualipilot.lambda_handler` for quick local smoke tests
if __name__ == "__main__":  # pragma: no cover
    import sys

    event = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(handler(event, None), indent=2, default=str))
