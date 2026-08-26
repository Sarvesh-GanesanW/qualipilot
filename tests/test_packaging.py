"""Static packaging contract tests."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_docker_version_defaults_match_package_version() -> None:
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        version = tomllib.load(project_file)["project"]["version"]

    for relative_path in ("docker/Dockerfile", "docker/Dockerfile.lambda"):
        dockerfile = (root / relative_path).read_text(encoding="utf-8")
        assert f"ARG VERSION={version}" in dockerfile
        assert 'org.opencontainers.image.version="${VERSION}"' in dockerfile
