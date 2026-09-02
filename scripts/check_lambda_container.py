"""Run the Lambda image through RIE against an ephemeral MinIO server.

This is an opt-in Docker gate, not a unit test and not an AWS deployment:

    python scripts/check_lambda_container.py

The gate builds ``docker/Dockerfile.lambda``, constrains the runtime container,
checks a real S3-backed invocation, then proves the input-size limit fails
before download. It prints one JSON evidence document and removes its tagged
image, containers, and network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "Dockerfile.lambda"
PLATFORM = "linux/amd64"
MINIO_IMAGE = (
    "quay.io/minio/minio@"
    "sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
)
MINIO_VERSION = "RELEASE.2025-09-07T16-13-09Z"
MINIO_USER = "minioadmin"
MINIO_PASSWORD = "minioadmin"
MINIO_KMS_KEY = "my-minio-key:OSMM+vkKUTCvQs9YL/CVMIMt43HFhkUpqJxTmGl6rYw="
LAMBDA_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_INPUT_BYTES = 1024
MAX_DATASET_BYTES = 8192
EXPECTED_SUMMARY = {
    "dataset_contract": "ok",
    "missing_values": "ok",
    "duplicates": "ok",
    "data_types": "ok",
    "outliers": "ok",
    "ranges": "ok",
    "cardinality": "ok",
}

SUCCESS_EVENT = {
    "s3_uri": "s3://quality/incoming/input.csv",
    "output_key": "reports/container-success.json",
    "fail_on": "none",
    "config": {
        "engine": "polars",
        "checks": {
            "required_columns": ["id", "amount", "quantity"],
            "expected_dtypes": {
                "id": "integer",
                "amount": "integer",
                "quantity": "integer",
            },
            "column_ranges": {
                "amount": {"min": 0, "max": 100},
                "quantity": {"min": 0, "max": 100},
            },
            "sample_size": 0,
            "include_top_values": False,
        },
        "llm": {"provider": "none"},
    },
}
LIMIT_EVENT = {
    "s3_uri": "s3://quality/incoming/oversized.csv",
    "output_key": "reports/oversized-should-not-exist.json",
    "fail_on": "none",
    "config": {"engine": "polars", "llm": {"provider": "none"}},
}

PROVISION_CODE = f"""
import json
import boto3

client = boto3.client("s3")
client.create_bucket(Bucket="quality")
objects = {{
    "incoming/input.csv": (
        b"id,amount,quantity\\n1,10,10\\n"
        b"2,20,20\\n3,30,30\\n"
    ),
    "incoming/oversized.csv": b"x" * {MAX_INPUT_BYTES + 1},
}}
for key, body in objects.items():
    client.put_object(Bucket="quality", Key=key, Body=body)
print(json.dumps({{
    key: client.head_object(Bucket="quality", Key=key)["ContentLength"]
    for key in objects
}}, sort_keys=True))
"""

VERIFY_CODE = """
import json
import boto3
from botocore.exceptions import ClientError

client = boto3.client("s3")
head = client.head_object(
    Bucket="quality", Key="reports/container-success.json"
)
body = json.loads(client.get_object(
    Bucket="quality", Key="reports/container-success.json"
)["Body"].read())
try:
    client.head_object(
        Bucket="quality", Key="reports/oversized-should-not-exist.json"
    )
except ClientError as exc:
    absent_code = exc.response["Error"]["Code"]
else:
    absent_code = None
print(json.dumps({
    "all_checks_completed": all(
        result["status"] == "completed" for result in body["results"]
    ),
    "check_names": sorted(result["name"] for result in body["results"]),
    "dataset_engine": body["dataset"]["engine"],
    "dataset_rows": body["dataset"]["row_count"],
    "dataset_source": body["dataset"]["source"],
    "metadata": head["Metadata"],
    "oversized_report_absent_code": absent_code,
    "report_bytes": head["ContentLength"],
    "server_side_encryption": head.get("ServerSideEncryption"),
}, sort_keys=True))
"""


def _run(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        detail = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}\n{detail}"
        )
    return completed


def _docker(
    *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return _run(["docker", *arguments], check=check)


def _parse_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} did not return JSON: {value!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} did not return a JSON object")
    return cast(dict[str, Any], parsed)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _s3_environment() -> list[str]:
    values = (
        f"AWS_ACCESS_KEY_ID={MINIO_USER}",
        f"AWS_SECRET_ACCESS_KEY={MINIO_PASSWORD}",
        "AWS_DEFAULT_REGION=us-east-1",
        "AWS_REGION=us-east-1",
        "AWS_ENDPOINT_URL_S3=http://minio:9000",
        "AWS_EC2_METADATA_DISABLED=true",
    )
    return [argument for value in values for argument in ("--env", value)]


def _inspect(target: str) -> dict[str, Any]:
    parsed: object = json.loads(_docker("inspect", target).stdout)
    if (
        not isinstance(parsed, list)
        or not parsed
        or not isinstance(parsed[0], dict)
    ):
        raise RuntimeError(
            f"docker inspect returned an invalid result for {target}"
        )
    return cast(dict[str, Any], parsed[0])


def _wait_for_healthy(container: str) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = cast(dict[str, Any], _inspect(container)["State"])
        health = cast(dict[str, Any], state.get("Health", {}))
        if health.get("Status") == "healthy":
            return
        if not state.get("Running"):
            logs = _docker("logs", container, check=False).stdout
            raise RuntimeError(f"MinIO exited before becoming healthy\n{logs}")
        time.sleep(0.25)
    logs = _docker("logs", container, check=False).stdout
    raise RuntimeError(
        f"MinIO did not become healthy within 45 seconds\n{logs}"
    )


def _container_python(
    image: str,
    network: str,
    code: str,
    label: str,
) -> dict[str, Any]:
    completed = _docker(
        "run",
        "--rm",
        "--network",
        network,
        "--cpus",
        "0.5",
        "--memory",
        "256m",
        "--memory-swap",
        "256m",
        "--pids-limit",
        "128",
        "--entrypoint",
        "python",
        *_s3_environment(),
        image,
        "-c",
        code,
    )
    return _parse_object(completed.stdout, label)


def _invoke(image: str, network: str, event: dict[str, Any]) -> dict[str, Any]:
    encoded_event = json.dumps(event, separators=(",", ":"))
    code = f"""
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

request = Request(
    "http://lambda-rie:8080/2015-03-31/functions/function/invocations",
    data={encoded_event!r}.encode(),
    headers={{"Content-Type": "application/json"}},
    method="POST",
)
deadline = time.monotonic() + 45
while True:
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=45) as response:
            result = {{
                "status_code": response.status,
                "wall_seconds": round(time.perf_counter() - started, 6),
                "payload": json.loads(response.read()),
            }}
        print(json.dumps(result, sort_keys=True))
        break
    except URLError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(0.1)
"""
    return _container_python(image, network, code, "Lambda RIE")


def _payload(invocation: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], invocation["payload"])


def _check_limits(container: dict[str, Any]) -> dict[str, Any]:
    host = cast(dict[str, Any], container["HostConfig"])
    config = cast(dict[str, Any], container["Config"])
    expected = {
        "nano_cpus": 1_000_000_000,
        "memory_bytes": LAMBDA_MEMORY_BYTES,
        "memory_swap_bytes": LAMBDA_MEMORY_BYTES,
        "pids_limit": 256,
        "read_only_root": True,
        "user": "1000:1000",
    }
    actual = {
        "nano_cpus": host["NanoCpus"],
        "memory_bytes": host["Memory"],
        "memory_swap_bytes": host["MemorySwap"],
        "pids_limit": host["PidsLimit"],
        "read_only_root": host["ReadonlyRootfs"],
        "user": config["User"],
    }
    _require(actual == expected, f"Lambda limits differ: {actual!r}")
    return actual


def _image_evidence(image: str) -> dict[str, Any]:
    details = _inspect(image)
    return {
        "dockerfile_sha256": hashlib.sha256(
            DOCKERFILE.read_bytes()
        ).hexdigest(),
        "image_id": details["Id"],
        "labels": cast(dict[str, str], details["Config"]["Labels"]),
        "platform": f"{details['Os']}/{details['Architecture']}",
    }


def _build_image(image: str) -> tuple[float, dict[str, Any]]:
    revision = _run(
        ["git", "rev-parse", "HEAD"],
        check=False,
    ).stdout.strip()
    started = time.perf_counter()
    _docker(
        "build",
        "--platform",
        PLATFORM,
        "--build-arg",
        f"VCS_REF={revision or 'unknown'}",
        "--file",
        str(DOCKERFILE),
        "--tag",
        image,
        ".",
    )
    elapsed = time.perf_counter() - started
    return elapsed, _image_evidence(image)


def _prepare_image(
    image: str, *, build: bool
) -> tuple[float | None, dict[str, Any]]:
    if build:
        return _build_image(image)
    return None, _image_evidence(image)


def _require_image_revision(
    evidence: dict[str, Any], expected_revision: str | None
) -> None:
    labels = cast(dict[str, str], evidence["labels"])
    actual = labels.get("org.opencontainers.image.revision")
    _require(actual not in {None, "", "unknown"}, "image revision is unpinned")
    if expected_revision is not None:
        _require(actual == expected_revision, "image revision differs")


def _start_minio(container: str, network: str) -> None:
    if _docker("image", "inspect", MINIO_IMAGE, check=False).returncode:
        _docker("pull", "--platform", PLATFORM, MINIO_IMAGE)
    _docker(
        "run",
        "--detach",
        "--name",
        container,
        "--network",
        network,
        "--network-alias",
        "minio",
        "--cpus",
        "0.5",
        "--memory",
        "256m",
        "--memory-swap",
        "256m",
        "--pids-limit",
        "128",
        "--tmpfs",
        "/data:rw,size=128m",
        "--health-cmd",
        "mc ready local",
        "--health-interval",
        "1s",
        "--health-timeout",
        "2s",
        "--health-retries",
        "30",
        "--env",
        f"MINIO_ROOT_USER={MINIO_USER}",
        "--env",
        f"MINIO_ROOT_PASSWORD={MINIO_PASSWORD}",
        "--env",
        f"MINIO_KMS_SECRET_KEY={MINIO_KMS_KEY}",
        MINIO_IMAGE,
        "server",
        "/data",
        "--address",
        ":9000",
        "--console-address",
        ":9001",
    )
    _wait_for_healthy(container)


def _start_lambda(container: str, network: str, image: str) -> None:
    _docker(
        "run",
        "--detach",
        "--name",
        container,
        "--network",
        network,
        "--network-alias",
        "lambda-rie",
        "--read-only",
        "--user",
        "1000:1000",
        "--cpus",
        "1",
        "--memory",
        "1024m",
        "--memory-swap",
        "1024m",
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m,uid=1000,gid=1000",
        *_s3_environment(),
        "--env",
        "AWS_LAMBDA_FUNCTION_NAME=qualipilot-container-gate",
        "--env",
        "AWS_LAMBDA_FUNCTION_MEMORY_SIZE=1024",
        "--env",
        f"QUALIPILOT_MAX_INPUT_BYTES={MAX_INPUT_BYTES}",
        "--env",
        f"QUALIPILOT_MAX_DATASET_BYTES={MAX_DATASET_BYTES}",
        "--env",
        "QUALIPILOT_LOG_LEVEL=WARNING",
        image,
    )


def _run_gate(
    image: str,
    network: str,
    minio: str,
    lambda_name: str,
    *,
    build_image: bool,
    expected_revision: str | None,
) -> dict[str, Any]:
    docker_server = _parse_object(
        _docker("version", "--format", "{{json .Server}}").stdout,
        "Docker version",
    )
    build_seconds, image_evidence = _prepare_image(image, build=build_image)
    _require_image_revision(image_evidence, expected_revision)
    _docker("network", "create", "--internal", network)
    _start_minio(minio, network)
    provisioned = _container_python(
        image, network, PROVISION_CODE, "S3 provisioner"
    )
    _require(
        provisioned
        == {
            "incoming/input.csv": 43,
            "incoming/oversized.csv": MAX_INPUT_BYTES + 1,
        },
        f"unexpected provisioned objects: {provisioned!r}",
    )

    _start_lambda(lambda_name, network, image)
    lambda_inspect = _inspect(lambda_name)
    limits = _check_limits(lambda_inspect)

    success = _invoke(image, network, SUCCESS_EVENT)
    success_payload = _payload(success)
    _require(
        success["status_code"] == 200, "successful RIE call was not HTTP 200"
    )
    _require(
        success_payload.get("cached") is False,
        "first call was unexpectedly cached",
    )
    _require(
        success_payload.get("bucket") == "quality", "response bucket differs"
    )
    _require(
        success_payload.get("input_key") == "incoming/input.csv",
        "response input key differs",
    )
    _require(
        success_payload.get("output_key") == "reports/container-success.json",
        "response output key differs",
    )
    _require(
        success_payload.get("summary") == EXPECTED_SUMMARY,
        "quality summary differs",
    )
    _require(
        success_payload.get("execution_failures") == 0,
        "quality execution failed",
    )
    _require(
        success_payload.get("llm_status") == "disabled", "LLM was not disabled"
    )

    cached = _invoke(image, network, SUCCESS_EVENT)
    _require(cached["status_code"] == 200, "cached RIE call was not HTTP 200")
    _require(
        _payload(cached).get("cached") is True,
        "second call did not use report metadata",
    )
    _require(
        _payload(cached).get("summary") == EXPECTED_SUMMARY,
        "cached quality summary differs",
    )

    rejected = _invoke(image, network, LIMIT_EVENT)
    rejected_payload = _payload(rejected)
    _require(
        rejected["status_code"] == 200, "failed RIE call was not HTTP 200"
    )
    _require(
        rejected_payload.get("errorType") == "ValueError",
        "limit failure type differs",
    )
    expected_error = (
        "s3://quality/incoming/oversized.csv is "
        f"{MAX_INPUT_BYTES + 1} bytes; limit is {MAX_INPUT_BYTES} bytes"
    )
    _require(
        rejected_payload.get("errorMessage") == expected_error,
        "limit failure message differs",
    )

    stored = _container_python(image, network, VERIFY_CODE, "S3 verifier")
    _require(
        stored.get("server_side_encryption") == "AES256",
        "report is not SSE-S3 encrypted",
    )
    _require(
        stored.get("dataset_rows") == 3, "stored report row count differs"
    )
    _require(
        stored.get("dataset_engine") == "polars",
        "stored report engine differs",
    )
    _require(
        stored.get("dataset_source") == "s3://quality/incoming/input.csv",
        "stored report source differs",
    )
    _require(
        stored.get("all_checks_completed") is True,
        "a stored check did not complete",
    )
    _require(
        stored.get("check_names") == sorted(EXPECTED_SUMMARY),
        "stored check set differs",
    )
    _require(
        stored.get("oversized_report_absent_code")
        in {"404", "NoSuchKey", "NotFound"},
        "oversized input unexpectedly produced a report",
    )
    metadata = cast(dict[str, str], stored.get("metadata"))
    _require(
        bool(metadata.get("qualipilot-identity")), "report identity is absent"
    )
    _require(
        json.loads(metadata.get("qualipilot-summary", "null"))
        == EXPECTED_SUMMARY,
        "report summary metadata differs",
    )
    _require(
        metadata.get("qualipilot-llm-status") == "disabled",
        "report LLM metadata differs",
    )
    _require(
        metadata.get("qualipilot-execution-failures") == "0",
        "report execution-failure metadata differs",
    )

    state = cast(dict[str, Any], _inspect(lambda_name)["State"])
    _require(state.get("Running") is True, "Lambda container stopped")
    _require(
        state.get("OOMKilled") is False, "Lambda container was OOM-killed"
    )
    stats = _parse_object(
        _docker(
            "stats", "--no-stream", "--format", "{{json .}}", lambda_name
        ).stdout,
        "Docker stats",
    )
    minio_version = _docker(
        "exec", minio, "minio", "--version"
    ).stdout.splitlines()[0]
    _require(MINIO_VERSION in minio_version, "running MinIO version differs")
    return {
        "status": "PASS",
        "boundaries": {
            "external_aws_service_calls": 0,
            "aws_services": "none; local Lambda RIE and MinIO only",
            "docker_network": "ephemeral internal bridge (no external route)",
            "s3_transport": "boto3 SigV4 over plain HTTP inside the bridge",
            "lambda_service_features": "not exercised",
        },
        "build": {
            **image_evidence,
            "seconds": (
                None if build_seconds is None else round(build_seconds, 6)
            ),
        },
        "runtime": {
            "docker_server": docker_server,
            "lambda_limits": limits,
            "lambda_stats_snapshot": stats,
            "minio_image": MINIO_IMAGE,
            "minio_version": minio_version,
            "rie_endpoint": "http://lambda-rie:8080 on internal bridge",
        },
        "success": {
            "first_invocation": success,
            "cached_invocation": cached,
            "stored_report": stored,
        },
        "input_limit": {
            "configured_bytes": MAX_INPUT_BYTES,
            "object_bytes": MAX_INPUT_BYTES + 1,
            "invocation": {
                "status_code": rejected["status_code"],
                "wall_seconds": rejected["wall_seconds"],
                "error_type": rejected_payload["errorType"],
                "error_message": rejected_payload["errorMessage"],
            },
            "report_created": False,
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        help="use an existing local Lambda image instead of building one",
    )
    parser.add_argument(
        "--expected-revision",
        help="require the image OCI revision label to match this value",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    suffix = uuid4().hex[:12]
    image = args.image or f"qualipilot-lambda-rie-gate:{suffix}"
    owns_image = args.image is None
    network = f"qualipilot-rie-{suffix}"
    minio = f"qualipilot-minio-{suffix}"
    lambda_name = f"qualipilot-lambda-{suffix}"
    print(
        f"local gate resources: {network}, {minio}, {lambda_name}",
        file=sys.stderr,
        flush=True,
    )
    try:
        evidence = _run_gate(
            image,
            network,
            minio,
            lambda_name,
            build_image=owns_image,
            expected_revision=args.expected_revision,
        )
    finally:
        _docker("rm", "--force", lambda_name, minio, check=False)
        _docker("network", "rm", network, check=False)
        if owns_image:
            _docker("image", "rm", image, check=False)
    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
